from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus import thgl_software


def test_thgl_fetch_requires_exact_license_and_writes_hash_receipt(tmp_path: Path) -> None:
    called = False

    def downloader(url: str, path: Path) -> None:
        nonlocal called
        called = True
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("thgl_software/readme.txt", "fixture")

    with pytest.raises(ContractViolation, match="CC-BY-4.0"):
        thgl_software.fetch_thgl_software(tmp_path, accept_license="unknown", downloader=downloader)
    assert called is False
    receipt = thgl_software.fetch_thgl_software(
        tmp_path, accept_license="CC-BY-4.0", downloader=downloader
    )
    assert len(receipt["archiveSha256"]) == 64
    assert receipt["formalEligible"] is False


def test_thgl_invalid_download_never_publishes_archive_or_receipt(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="valid ZIP"):
        thgl_software.fetch_thgl_software(
            tmp_path,
            accept_license="CC-BY-4.0",
            downloader=lambda _url, path: path.write_bytes(b"truncated-not-a-zip"),
        )
    raw = tmp_path / "datasets/raw/gfm/thgl-software"
    assert not (raw / "thgl-software-2.0.0.zip").exists()
    assert not (raw / "fetch-receipt.json").exists()
    assert not list(raw.glob(".thgl-software-2.0.0.zip.*.tmp"))


def test_thgl_malicious_zip_never_publishes_archive_or_receipt(tmp_path: Path) -> None:
    def malicious(_url: str, path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../escape.txt", "must-never-be-published")

    with pytest.raises(ContractViolation, match="unsafe path"):
        thgl_software.fetch_thgl_software(
            tmp_path,
            accept_license="CC-BY-4.0",
            downloader=malicious,
        )
    raw = tmp_path / "datasets/raw/gfm/thgl-software"
    assert not (raw / "thgl-software-2.0.0.zip").exists()
    assert not (raw / "fetch-receipt.json").exists()
    assert not list(raw.glob(".thgl-software-2.0.0.zip.*.tmp"))


def test_thgl_existing_archive_is_revalidated_and_not_overwritten(tmp_path: Path) -> None:
    calls = 0

    def valid(_url: str, path: Path) -> None:
        nonlocal calls
        calls += 1
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("thgl_software/readme.txt", "fixture")

    first = thgl_software.fetch_thgl_software(
        tmp_path, accept_license="CC-BY-4.0", downloader=valid
    )
    archive = tmp_path / "datasets/raw/gfm/thgl-software/thgl-software-2.0.0.zip"
    before = archive.read_bytes()
    second = thgl_software.fetch_thgl_software(
        tmp_path, accept_license="CC-BY-4.0", downloader=valid
    )
    assert calls == 1
    assert archive.read_bytes() == before
    assert second["archiveSha256"] == first["archiveSha256"]


def test_thgl_formal_fetch_rejects_official_object_metadata_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractViolation, match="size/ETag drifted"):
        thgl_software.fetch_thgl_software(
            tmp_path,
            accept_license="CC-BY-4.0",
            metadata_client=lambda _: {"content-length": "1", "etag": "changed"},
        )


def test_thgl_runtime_array_contract_and_fixture_prepare(tmp_path: Path) -> None:
    def downloader(url: str, path: Path) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("thgl_software/data.csv", "fixture")

    thgl_software.fetch_thgl_software(
        tmp_path, accept_license="CC-BY-4.0", downloader=downloader
    )
    data = SimpleNamespace(
        full_data={
            "sources": np.arange(14, dtype=np.int64) % 4,
            "destinations": (np.arange(14, dtype=np.int64) + 1) % 4,
            "timestamps": np.arange(14, dtype=np.int64),
            "edge_type": np.arange(14, dtype=np.int64),
        },
        node_type=np.arange(4, dtype=np.int64),
        train_mask=np.asarray([True] * 10 + [False] * 4),
        val_mask=np.asarray([False] * 10 + [True] * 2 + [False] * 2),
        test_mask=np.asarray([False] * 12 + [True] * 2),
    )
    archive = tmp_path / "datasets/raw/gfm/thgl-software/thgl-software-2.0.0.zip"
    manifest = thgl_software.prepare_thgl_software(
        archive,
        tmp_path,
        dataset_factory=lambda _: data,
        enforce_official_counts=False,
    )
    assert manifest["nodeTypeCount"] == 4
    assert manifest["relationCount"] == 14
    assert manifest["splits"]["strategy"] == "official-temporal-70-15-15"
    assert manifest["splits"]["counts"] == {"train": 10, "validation": 2, "test": 2}
    assert manifest["splits"]["trainEndInclusive"] == 9
    assert manifest["splits"]["validationStartInclusive"] == 10
    assert manifest["splits"]["validationEndInclusive"] == 11
    assert manifest["splits"]["testStartInclusive"] == 12
    splits = thgl_software.load_thgl_software_splits(tmp_path)
    assert splits["indices"]["train"].tolist() == list(range(10))
    assert splits["indices"]["validation"].tolist() == [10, 11]
    assert splits["indices"]["test"].tolist() == [12, 13]
    assert splits["bounds"] == {
        "trainEndInclusive": 9,
        "validationStartInclusive": 10,
        "validationEndInclusive": 11,
        "testStartInclusive": 12,
    }


def test_thgl_masks_reject_temporal_overlap_and_nonempty_violations() -> None:
    base = {
        "sources": np.arange(14, dtype=np.int64) % 4,
        "destinations": (np.arange(14, dtype=np.int64) + 1) % 4,
        "timestamps": np.arange(14, dtype=np.int64),
        "edge_type": np.arange(14, dtype=np.int64),
    }
    overlapping = SimpleNamespace(
        full_data=base,
        node_type=np.arange(4, dtype=np.int64),
        train_mask=np.asarray([True] * 10 + [False] * 4),
        val_mask=np.asarray([False] * 9 + [True] * 3 + [False] * 2),
        test_mask=np.asarray([False] * 12 + [True] * 2),
    )
    with pytest.raises(ContractViolation, match="disjoint|temporal ordering"):
        thgl_software._runtime_arrays(overlapping)

    empty_validation = SimpleNamespace(
        full_data=base,
        node_type=np.arange(4, dtype=np.int64),
        train_mask=np.asarray([True] * 12 + [False] * 2),
        val_mask=np.zeros(14, dtype=np.bool_),
        test_mask=np.asarray([False] * 12 + [True] * 2),
    )
    with pytest.raises(ContractViolation, match="at least one event"):
        thgl_software._runtime_arrays(empty_validation)


def test_default_prepare_uses_fresh_isolation_and_never_reads_raw_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def downloader(url: str, path: Path) -> None:
        edges = ["timestamp,head,tail,relation"]
        edges.extend(
            f"{index},{100 + index % 4},{100 + (index + 1) % 4},{index % 14}"
            for index in range(20)
        )
        node_types = ["node_id,type", "100,0", "101,1", "102,2", "103,3"]
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("thgl-software_edgelist.csv", "\n".join(edges))
            archive.writestr("thgl-software_nodetype.csv", "\n".join(node_types))

    thgl_software.fetch_thgl_software(
        tmp_path, accept_license="CC-BY-4.0", downloader=downloader
    )
    raw = tmp_path / "datasets/raw/gfm/thgl-software"
    malicious = raw / "existing-malicious.pkl"
    malicious.write_bytes(b"do-not-unpickle-this-sentinel")
    captured: list[Path] = []

    real_parser = thgl_software._parse_official_csv

    def parse(dataset_root: Path) -> dict[str, np.ndarray]:
        isolated_root = dataset_root.parent
        captured.append(isolated_root)
        assert isolated_root.parent == (tmp_path / "tmp").resolve()
        assert isolated_root != raw.resolve()
        assert (dataset_root / "thgl-software_edgelist.csv").is_file()
        assert not (dataset_root / malicious.name).exists()
        (dataset_root / "ml_thgl-software.pkl").write_bytes(b"temporary-tgb-cache")
        return real_parser(dataset_root)

    monkeypatch.setattr(thgl_software, "_parse_official_csv", parse)
    archive = raw / "thgl-software-2.0.0.zip"
    thgl_software.prepare_thgl_software(
        archive,
        tmp_path,
        enforce_official_counts=False,
    )
    assert len(captured) == 1
    assert not captured[0].exists()
    assert malicious.read_bytes() == b"do-not-unpickle-this-sentinel"
    processed = tmp_path / "datasets/processed/gfm/thgl-software-2.0.0"
    assert not list(processed.rglob("*.pkl"))


def test_direct_csv_adapter_reproduces_first_seen_node_mapping_and_quantile_split(
    tmp_path: Path,
) -> None:
    root = tmp_path / "thgl_software"
    root.mkdir()
    edges = ["timestamp,head,tail,relation"]
    edges.extend(
        f"{index},{(90, 10, 40, 20)[index % 4]},{(10, 40, 20, 90)[index % 4]},{index % 14}"
        for index in range(20)
    )
    (root / "thgl-software_edgelist.csv").write_text(
        "\n".join(edges), encoding="utf-8"
    )
    (root / "thgl-software_nodetype.csv").write_text(
        "node_id,type\n90,3\n10,2\n40,1\n20,0", encoding="utf-8"
    )
    arrays = thgl_software._parse_official_csv(root)
    assert arrays["src"][:4].tolist() == [0, 1, 2, 3]
    assert arrays["dst"][:4].tolist() == [1, 2, 3, 0]
    assert arrays["node_type"].tolist() == [3, 2, 1, 0]
    assert arrays["train_mask"].sum() == 14
    assert arrays["validation_mask"].sum() == 3
    assert arrays["test_mask"].sum() == 3
