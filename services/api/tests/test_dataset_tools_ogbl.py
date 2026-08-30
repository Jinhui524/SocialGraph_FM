from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.dataset_imports import DatasetImportService
from app.dataset_tools import convert_ogbl_collab


class _FakeData:
    def __init__(self, *, message_years: list[int] | None = None) -> None:
        self.edge_index = np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
        )
        self.x = np.eye(6, 128, dtype=np.float32)
        self.edge_weight = np.asarray([1, 1, 2, 2], dtype=np.float32)
        self.edge_year = np.asarray(
            message_years or [2016, 2016, 2017, 2017], dtype=np.int64
        )


class _FakeDataset:
    def __init__(self, *, message_years: list[int] | None = None) -> None:
        self._data = _FakeData(message_years=message_years)

    def __getitem__(self, index: int) -> _FakeData:
        assert index == 0
        return self._data

    def get_edge_split(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "train": {"edge": np.asarray([[0, 1], [1, 2]], dtype=np.int64)},
            "valid": {
                "edge": np.asarray([[2, 3]], dtype=np.int64),
                "edge_neg": np.asarray([[0, 5]], dtype=np.int64),
            },
            "test": {
                "edge": np.asarray([[3, 4]], dtype=np.int64),
                "edge_neg": np.asarray([[1, 5]], dtype=np.int64),
            },
        }


def _install_fake_ogb(
    monkeypatch: pytest.MonkeyPatch,
    *,
    message_years: list[int] | None = None,
) -> None:
    ogb = types.ModuleType("ogb")
    ogb.__version__ = "1.3.6"
    linkproppred = types.ModuleType("ogb.linkproppred")

    def factory(*, name: str, root: str) -> _FakeDataset:
        assert name == "ogbl-collab"
        assert root
        return _FakeDataset(message_years=message_years)

    linkproppred.PygLinkPropPredDataset = factory  # type: ignore[attr-defined]
    ogb.linkproppred = linkproppred  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ogb", ogb)
    monkeypatch.setitem(sys.modules, "ogb.linkproppred", linkproppred)


def _cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "ogb"
    processed = root / "ogbl_collab" / "processed"
    processed.mkdir(parents=True)
    (processed / "geometric_data_processed.pt").write_bytes(b"trusted-test-marker")
    return root


def _raw_cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "ogb"
    dataset = root / "ogbl_collab"
    for relative in (
        "RELEASE_v1.txt",
        "raw/edge.csv.gz",
        "raw/edge_weight.csv.gz",
        "raw/edge_year.csv.gz",
        "raw/node-feat.csv.gz",
        "raw/num-edge-list.csv.gz",
        "raw/num-node-list.csv.gz",
        "split/time/train.pt",
        "split/time/valid.pt",
        "split/time/test.pt",
    ):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"trusted-pinned-release-test-marker")
    return root


def test_ogbl_adapter_is_offline_and_requires_a_complete_local_cache(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="必须显式 --trust-pickle"):
        convert_ogbl_collab(tmp_path, tmp_path / "never-created.zip", trust_pickle=False)

    empty_root = tmp_path / "empty"
    (empty_root / "ogbl_collab").mkdir(parents=True)
    with pytest.raises(ValueError, match="适配器禁止联网下载"):
        convert_ogbl_collab(empty_root, tmp_path / "still-not-created.zip", trust_pickle=True)


def test_ogbl_adapter_accepts_a_complete_release_v1_raw_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ogb(monkeypatch)
    source = _raw_cache_root(tmp_path)
    package = tmp_path / "raw-cache.sgfm.zip"

    manifest = convert_ogbl_collab(source, package, trust_pickle=True)

    assert package.is_file()
    assert manifest["datasets"][0]["name"] == "ogbl-collab"


def test_ogbl_safe_package_is_path_and_time_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ogb(monkeypatch)
    first_source = _cache_root(tmp_path / "first-location")
    second_source = _cache_root(tmp_path / "second-location")
    first_package = tmp_path / "first.sgfm.zip"
    second_package = tmp_path / "second.sgfm.zip"

    convert_ogbl_collab(first_source, first_package, trust_pickle=True)
    convert_ogbl_collab(second_source, second_package, trust_pickle=True)

    assert first_package.read_bytes() == second_package.read_bytes()


def test_ogbl_adapter_package_round_trips_through_readiness_and_materializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ogb(monkeypatch)
    source = _cache_root(tmp_path)
    package = tmp_path / "ogbl-collab.sgfm.zip"

    manifest = convert_ogbl_collab(source, package, trust_pickle=True)

    item = manifest["datasets"][0]
    assert manifest["trustedSource"] == "local-official-ogbl-collab-cache"
    assert item["name"] == "ogbl-collab"
    assert item["datasetRole"] == "benchmark"
    assert item["licensePolicy"]["status"] == "verified"
    assert item["licensePolicy"]["identifier"] == "ODC-BY-1.0"
    assert item["linkPredictionProtocol"]["trainYearMax"] == 2017
    assert item["linkPredictionProtocol"]["validationYear"] == 2018
    assert item["linkPredictionProtocol"]["testYear"] == 2019
    assert item["linkPredictionProtocol"]["evaluator"] == (
        "ogb.linkproppred.Evaluator(ogbl-collab)"
    )
    assert item["linkPredictionProtocol"]["evaluatorVersion"] == "1.3.6"

    with zipfile.ZipFile(package) as archive:
        archived_manifest = json.loads(archive.read("manifest.json"))
        assert archived_manifest["sourceFingerprint"] == manifest["sourceFingerprint"]
        with (
            archive.open("datasets/ogbl-collab/graph.npz") as graph_file,
            np.load(graph_file, allow_pickle=False) as arrays,
        ):
            assert arrays["x"].shape == (6, 128)
            assert arrays["edge_index"].shape == (2, 4)
            assert arrays["edge_timestamp"].dtype == np.dtype(np.int16)
            assert arrays["variant_validation_negative"].shape == (2, 1)

    service = DatasetImportService(
        Settings(dataset_storage_root=str(tmp_path / "isolated-store"))
    )
    artifact = service.import_trusted_package(
        str(package), job_id="job-ogbl-adapter", source_path=str(source)
    )[0]
    assert artifact.schema_version == "2.2"
    assert artifact.training_ref is not None
    readiness = service.readiness(
        artifact.id, training_ref_hash=artifact.training_ref.ref_hash
    )
    assert readiness.status == "ready", readiness.blockers
    bundle = service.materialize_contract(
        artifact.id, training_ref_hash=artifact.training_ref.ref_hash
    )
    assert bundle.feature_shape == [6, 128]
    assert bundle.split_sizes == {"train": 2, "validation": 1, "test": 1}


def test_ogbl_adapter_rejects_future_edges_in_message_passing_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_ogb(monkeypatch, message_years=[2016, 2016, 2018, 2018])
    source = _cache_root(tmp_path)

    with pytest.raises(ValueError, match="存在时间泄漏"):
        convert_ogbl_collab(
            source,
            tmp_path / "future-leak.sgfm.zip",
            trust_pickle=True,
        )
