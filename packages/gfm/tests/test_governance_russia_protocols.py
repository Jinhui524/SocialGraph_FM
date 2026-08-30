from __future__ import annotations

from pathlib import Path

import pytest

from socialgraph_gfm.governance.cli import main
from socialgraph_gfm.governance.russia_protocols import (
    EXPECTED_FOCUS_COUNTS,
    PROTOCOL_DIRECTORIES,
    RussiaProtocolFocusManifest,
    RussiaProtocolFocusProjection,
    generate_russia_protocol_focus,
    verify_russia_protocol_focus,
)


def _global_model_root() -> Path:
    repository = Path(__file__).resolve().parents[3]
    root = repository / "var" / "gfm" / "global-model"
    if not (root / "registry" / "socialgraph-global.json").is_file():
        pytest.skip("the ignored frozen SocialGraph-FM Global release is not installed")
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_protocol_focus_is_byte_deterministic_and_provenance_bound(tmp_path: Path) -> None:
    global_model_root = _global_model_root()
    first = generate_russia_protocol_focus(global_model_root, tmp_path / "first")
    second = generate_russia_protocol_focus(global_model_root, tmp_path / "second")
    assert _files(first) == _files(second)
    assert set(_files(first)) == {
        f"{directory}/{name}"
        for directory, _protocol in PROTOCOL_DIRECTORIES
        for name in ("manifest.json", "projection.json")
    }
    assert not list(first.rglob("*.zip"))

    training_hashes: set[str] = set()
    expected_train_nodes = {"in_domain": 429, "low_label": 23, "cross_domain": 43_301, "global": 43_730}
    for directory, protocol in PROTOCOL_DIRECTORIES:
        projection = RussiaProtocolFocusProjection.model_validate_json(
            (first / directory / "projection.json").read_bytes(), strict=True
        )
        manifest = RussiaProtocolFocusManifest.model_validate_json(
            (first / directory / "manifest.json").read_bytes(), strict=True
        )
        assert projection.protocol == protocol
        assert projection.directory_protocol == directory
        assert (projection.node_count, projection.edge_count) == EXPECTED_FOCUS_COUNTS[protocol]
        assert sum(node.is_anchor for node in projection.nodes) == 40
        assert [node.anchor_rank for node in projection.nodes if node.is_anchor]
        assert projection.projection_only is True
        assert projection.uploadable is False
        assert projection.inference_required is False
        assert projection.selection_uses_labels is False
        assert projection.metric_scope == "none-projection-is-not-an-evaluation-sample"
        assert projection.provenance.evaluation_split_hash == (
            "6ad19d77cc5c5ed838bcafcd22c3abbd9698c35e58650c76fe9c3aa80679f643"
        )
        assert projection.provenance.labelled_train_nodes == expected_train_nodes[protocol]
        training_hashes.add(projection.provenance.training_hash)
        assert manifest.projection_hash == projection.projection_hash
        assert manifest.provenance == projection.provenance
        assert all(
            int(edge.source.partition(":")[2]) < int(edge.target.partition(":")[2])
            for edge in projection.edges
        )
        assert all(edge.modalities for edge in projection.edges)

    assert len(training_hashes) == 4

    global_projection = RussiaProtocolFocusProjection.model_validate_json(
        (first / "global" / "projection.json").read_bytes(), strict=True
    )
    assert global_projection.protocol == "global"
    assert global_projection.directory_protocol == "global"
    assert tuple(item.manifest_hash for item in verify_russia_protocol_focus(global_model_root, first))


def test_protocol_focus_cli_and_tamper_verification(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    global_model_root = _global_model_root()
    destination = tmp_path / "protocols"
    assert (
        main(
            [
                "russia-protocol-focus",
                "--governance-root",
                str(global_model_root),
                "--output-dir",
                str(destination),
            ]
        )
        == 0
    )
    assert destination.as_posix().lower() in capsys.readouterr().out.replace("\\", "/").lower()
    assert (
        main(
            [
                "verify-russia-protocol-focus",
                "--governance-root",
                str(global_model_root),
                "--output-dir",
                str(destination),
            ]
        )
        == 0
    )
    assert len(capsys.readouterr().out.strip()) == 64

    projection_path = destination / "low_label" / "projection.json"
    projection_path.write_bytes(projection_path.read_bytes().replace(b'"nodeCount":53', b'"nodeCount":54'))
    with pytest.raises(ValueError):
        verify_russia_protocol_focus(global_model_root, destination)
