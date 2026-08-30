from __future__ import annotations

import inspect
import hashlib
import io
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from socialgraph_gfm.core.datasets.parsers import parse_facebook100_mat
from socialgraph_gfm.core.datasets.penn94_conversion import (
    PENN94_CONVERTER_VERSION,
    PENN94_DATA_SHA256,
    PENN94_LABELED_NODE_COUNT,
    PENN94_RAW_SPLIT_SHA256,
    convert_penn94_official_splits,
    load_penn94_safe_splits,
    validate_penn94_safe_splits,
    verify_penn94_raw_split,
    write_deterministic_safe_splits,
)
from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.datasets.recipes import load_dataset_recipes
import socialgraph_gfm.core.datasets.penn94_conversion as conversion
from socialgraph_gfm.errors import ArtifactRootNotConfigured


def _official_shape_arrays() -> dict[str, np.ndarray]:
    labeled = np.arange(PENN94_LABELED_NODE_COUNT, dtype=np.int64)
    roles = {"train": [], "valid": [], "test": []}
    for split_index in range(5):
        rotated = np.roll(labeled, split_index * 137)
        roles["train"].append(rotated[:19_407])
        roles["valid"].append(rotated[19_407:29_110])
        roles["test"].append(rotated[29_110:])
    return {name: np.stack(rows) for name, rows in roles.items()}


def test_converter_has_no_user_controlled_path_or_url_parameters() -> None:
    assert inspect.signature(convert_penn94_official_splits).parameters == {}


def test_converter_runtime_paths_are_derived_from_socialgraph_home(monkeypatch, tmp_path) -> None:
    home = tmp_path / "var" / "gfm"
    monkeypatch.setenv("SOCIALGRAPH_FM_HOME", str(home))

    paths = conversion._runtime_paths()

    assert paths.root == (home / "core-runtime").resolve()
    assert paths.raw_split == paths.root / "raw" / "facebook100" / "1.0.0" / "fb100-Penn94-splits.npy"
    assert paths.published_target.is_relative_to(paths.root)


def test_converter_runtime_paths_fail_closed_without_socialgraph_home(monkeypatch) -> None:
    monkeypatch.delenv("SOCIALGRAPH_FM_HOME", raising=False)

    with pytest.raises(ArtifactRootNotConfigured, match="SOCIALGRAPH_FM_HOME"):
        conversion._runtime_paths()


def test_wrong_sha_and_arbitrary_pickle_are_rejected_before_deserialization(tmp_path) -> None:
    wrong = tmp_path / "fb100-Penn94-splits.npy"
    wrong.write_bytes(pickle.dumps(os.system))

    with pytest.raises(ValueError, match="fixed raw SHA-256"):
        verify_penn94_raw_split(wrong)

    assert PENN94_RAW_SPLIT_SHA256 == "88a1060358482d8e25b978ab59c4ff71771388cd5ffac3dd775a3cd9dc85b032"


def test_hash_and_parse_use_one_immutable_byte_buffer(tmp_path, monkeypatch) -> None:
    raw = tmp_path / "fixed.npy"
    raw.write_bytes(b"not-even-a-valid-npy")
    calls = 0
    original = raw.read_bytes()

    real_open = type(raw).open

    def alternating_open(self, *args, **kwargs):
        nonlocal calls
        if self == raw and args and args[0] == "rb":
            calls += 1
            return io.BytesIO(
                original if calls == 1 else b"substituted-after-hash"
            )
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(conversion, "PENN94_RAW_SPLIT_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(type(raw), "open", alternating_open)
    with pytest.raises(ValueError):
        conversion._read_verified_object_payload(raw)
    assert calls == 1


def test_oversized_penn_raw_is_rejected_before_unbounded_buffering(
    tmp_path, monkeypatch
) -> None:
    maximum = {
        source.source_id: source.max_bytes
        for source in load_dataset_recipes()["facebook100"].sources
    }["Penn94-official-splits"]
    raw = tmp_path / "oversized.npy"
    with raw.open("wb") as stream:
        stream.seek(maximum)
        stream.write(b"x")
    read_bytes_calls = 0

    def forbidden_read_bytes(_self) -> bytes:
        nonlocal read_bytes_calls
        read_bytes_calls += 1
        raise AssertionError("unbounded Path.read_bytes reached oversized fixed asset")

    monkeypatch.setattr(type(raw), "read_bytes", forbidden_read_bytes)
    with pytest.raises(ValueError, match="maximum"):
        conversion._read_verified_object_payload(raw)
    assert read_bytes_calls == 0


def test_production_loader_rejects_pickle_backed_object_npy(tmp_path) -> None:
    raw = tmp_path / "object.npy"
    np.save(raw, np.array([{"train": np.array([0])}], dtype=object))

    with pytest.raises(ValueError, match="pickle|object"):
        load_penn94_safe_splits(
            raw,
            labeled_node_indices=np.arange(PENN94_LABELED_NODE_COUNT, dtype=np.int64),
        )


def test_safe_derived_asset_loads_in_fresh_process_without_pickle(tmp_path) -> None:
    output = tmp_path / "penn94-official-splits-safe.npz"
    arrays = _official_shape_arrays()
    first_sha = write_deterministic_safe_splits(output, arrays)
    second = tmp_path / "second.npz"
    second_sha = write_deterministic_safe_splits(second, arrays)
    assert first_sha == second_sha
    assert output.read_bytes() == second.read_bytes()

    command = [
        sys.executable,
        "-I",
        "-c",
        (
            "import numpy as np,sys; "
            "z=np.load(sys.argv[1],allow_pickle=False); "
            "assert set(z.files)=={'train','valid','test'}; "
            "assert z['train'].shape==(5,19407); "
            "assert z['valid'].shape==(5,9703); "
            "assert z['test'].shape==(5,9705); "
            "assert all(z[k].dtype.kind in 'iu' for k in z.files)"
        ),
        str(output),
    ]
    subprocess.run(command, check=True, timeout=30)
    splits = load_penn94_safe_splits(
        output,
        labeled_node_indices=np.arange(PENN94_LABELED_NODE_COUNT, dtype=np.int64),
    )
    assert len(splits) == 5
    assert tuple((len(s.train), len(s.validation), len(s.test)) for s in splits) == (
        (19_407, 9_703, 9_705),
    ) * 5


def test_facebook_parser_consumes_only_safe_derived_integer_splits(tmp_path) -> None:
    safe_splits = tmp_path / "penn94-official-splits-safe.npz"
    write_deterministic_safe_splits(safe_splits, _official_shape_arrays())
    profile = np.zeros((PENN94_LABELED_NODE_COUNT, 7), dtype=np.int64)
    profile[:, 1] = 1
    mat_path = tmp_path / "Penn94.mat"
    savemat(
        mat_path,
        {
            "A": sparse.csr_matrix(
                (PENN94_LABELED_NODE_COUNT, PENN94_LABELED_NODE_COUNT), dtype=np.uint8
            ),
            "local_info": profile,
        },
    )

    graph = parse_facebook100_mat(
        mat_path, graph_id="Penn94", official_splits_path=safe_splits
    )

    assert len(graph.official_splits) == 5
    assert all(len(split.train) == 19_407 for split in graph.official_splits)


@pytest.mark.parametrize(
    "mutation",
    [
        "float-dtype",
        "overlap",
        "outside-label-mask",
        "wrong-count",
    ],
)
def test_safe_split_validation_is_exact_and_fail_closed(mutation: str) -> None:
    arrays = _official_shape_arrays()
    labeled = np.arange(PENN94_LABELED_NODE_COUNT, dtype=np.int64)
    if mutation == "float-dtype":
        arrays["train"] = arrays["train"].astype(np.float64)
    elif mutation == "overlap":
        arrays["valid"][0, 0] = arrays["train"][0, 0]
    elif mutation == "outside-label-mask":
        labeled = labeled + 1
    else:
        arrays["test"] = arrays["test"][:, :-1]

    with pytest.raises(ValueError):
        validate_penn94_safe_splits(arrays, labeled_node_indices=labeled)


def test_existing_publication_rejects_self_consistent_rewritten_provenance(tmp_path) -> None:
    target = tmp_path / "published"
    target.mkdir()
    asset = target / "penn94-official-splits-safe.npz"
    derived_sha = write_deterministic_safe_splits(asset, _official_shape_arrays())
    recipe = load_dataset_recipes()["facebook100"]
    penn_url = {source.source_id: source.url for source in recipe.sources}["Penn94"]
    without_hash = {
        "schemaVersion": "socialgraph-fm.core-penn94-split-conversion/1.0",
        "sourceCommit": conversion.PENN94_LINKX_COMMIT,
        "sourceUrl": "https://attacker.example/repacked.npy",
        "sourceSha256": conversion.PENN94_RAW_SPLIT_SHA256,
        "penn94DataUrl": penn_url,
        "penn94DataObservedSha256": PENN94_DATA_SHA256,
        "derivedFormat": "npz with primitive little-endian int64 NPY members",
        "derivedSha256": derived_sha,
        "converterVersion": PENN94_CONVERTER_VERSION,
        "converterCodeSha256": conversion._converter_code_sha256(),
        "splitCount": 5,
        "labeledNodeCount": PENN94_LABELED_NODE_COUNT,
        "roleCounts": conversion.PENN94_SPLIT_COUNTS,
        "recipeSha256": recipe.recipe_sha256,
    }
    manifest = {**without_hash, "manifestSha256": canonical_sha256(without_hash)}
    (target / "conversion-manifest.json").write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fixed provenance"):
        conversion._validate_existing_publication(
            target, np.arange(PENN94_LABELED_NODE_COUNT, dtype=np.int64)
        )
