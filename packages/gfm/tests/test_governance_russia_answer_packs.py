from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.governance.russia_answer_packs import (
    ANSWER_PACK_FILENAMES,
    ANSWER_PACK_MAX_FUSED_EDGES,
    RussiaAnswerPackCatalog,
    generate_russia_answer_packs,
    verify_russia_answer_pack_catalog,
)


def _source_bundle() -> Path:
    repository = Path(__file__).resolve().parents[3]
    source = repository / "var" / "gfm" / "governance" / "samples" / "russia-replay.zip"
    if not source.is_file():
        pytest.skip("the ignored canonical Russia replay bundle is not installed")
    return source


def test_answer_packs_are_small_contract_valid_and_byte_deterministic(tmp_path: Path) -> None:
    source = _source_bundle()
    scores_path = source.parent.parent.parent / "global-model" / "exports" / "governance-socialgraph-fm-global/test" / "results" / "global-russia.npz"
    scores = None
    if scores_path.is_file():
        with np.load(scores_path, allow_pickle=False) as archive:
            scores = np.asarray(archive["scores"], dtype=np.float32)
    first_path = generate_russia_answer_packs(source, tmp_path / "first", frozen_scores=scores)
    second_path = generate_russia_answer_packs(source, tmp_path / "second", frozen_scores=scores)
    first = RussiaAnswerPackCatalog.model_validate_json(first_path.read_bytes())
    second = RussiaAnswerPackCatalog.model_validate_json(second_path.read_bytes())
    assert first.catalog_hash == second.catalog_hash
    assert tuple(item.file_name for item in first.packs) == ANSWER_PACK_FILENAMES
    for left, right in zip(first.packs, second.packs, strict=True):
        left_bytes = (first_path.parent / left.file_name).read_bytes()
        right_bytes = (second_path.parent / right.file_name).read_bytes()
        assert left_bytes == right_bytes
        assert hashlib.sha256(left_bytes).hexdigest() == left.sha256
        assert 72 <= left.node_count <= 144
        assert left.node_count < 183
        assert left.fused_undirected_edge_count <= ANSWER_PACK_MAX_FUSED_EDGES
        with zipfile.ZipFile(first_path.parent / left.file_name) as archive:
            assert archive.namelist() == [
                "manifest.json",
                "nodes.csv",
                "relations.csv",
                "features.npz",
            ]
            manifest = archive.read("manifest.json")
            assert b"label" not in manifest
            assert b"split" not in manifest
            assert b"score" not in manifest
            with np.load(archive.open("features.npz"), allow_pickle=False) as features:
                assert set(features.files) == {"node_ids", "text_features"}
                assert features["text_features"].shape == (left.node_count, 768)
                assert features["text_features"].dtype == np.dtype(np.float32)
    verified = verify_russia_answer_pack_catalog(source, first_path)
    assert verified.catalog_hash == first.catalog_hash
