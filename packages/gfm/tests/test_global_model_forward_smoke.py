from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.global_model import cli
from socialgraph_gfm.global_model.forward_smoke import (
    FORWARD_SMOKE_PROTOCOLS,
    FORWARD_SMOKE_SCHEMA,
    NEIGHBOR_FANOUT,
    NEIGHBOR_HOPS,
    SEED_NODE_COUNT,
    _bounded_russia_batch,
    _load_russia,
    _read_export,
    _safe_checkpoint,
    run_checkpoint_forward_smoke,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_MODEL_ROOT = REPOSITORY_ROOT / "bundles" / "models" / "socialgraph-global"


def _bundle_snapshot() -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(BUNDLED_MODEL_ROOT).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            file_sha256(path),
        )
        for path in BUNDLED_MODEL_ROOT.rglob("*")
        if path.is_file()
    }


def test_real_cpu_forward_smoke_covers_all_published_protocol_checkpoints() -> None:
    before = _bundle_snapshot()

    report = run_checkpoint_forward_smoke(BUNDLED_MODEL_ROOT, device="cpu")

    assert report["schemaVersion"] == FORWARD_SMOKE_SCHEMA
    assert report["passed"] is True
    assert report["readOnly"] is True
    assert report["device"] == "cpu"
    assert report["corpus"]["country"] == "russia"
    assert report["corpus"]["nodeCount"] == 716
    assert report["batch"]["seedNodeCount"] == SEED_NODE_COUNT
    assert report["batch"]["neighborFanout"] == NEIGHBOR_FANOUT
    assert report["batch"]["neighborHops"] == NEIGHBOR_HOPS
    assert report["batch"]["sampledNodeCount"] <= report["batch"]["maximumNodeCount"]
    assert report["protocolCount"] == 4
    assert tuple(item["protocol"] for item in report["protocols"]) == FORWARD_SMOKE_PROTOCOLS
    for item in report["protocols"]:
        assert item["checkpoint"]["relativePath"].endswith(f"checkpoints/{item['protocol']}.pt")
        assert len(item["checkpoint"]["sha256"]) == 64
        assert len(item["model"]["modelStateHash"]) == 64
        assert item["device"] == "cpu"
        assert item["batch"]["sampledNodeCount"] == report["batch"]["sampledNodeCount"]
        assert item["finite"] is True
        assert item["modelStateUnchanged"] is True
        assert item["router"]["routesAllowed"] is True
        assert item["router"]["weightsValid"] is True
        assert item["modalityContributionsValid"] is True
        assert item["shape"] == {
            "fusedFeatures": [report["batch"]["sampledNodeCount"], 256],
            "logits": [report["batch"]["sampledNodeCount"]],
            "modalityContributions": [report["batch"]["sampledNodeCount"], 2],
            "nodeEmbeddings": [report["batch"]["sampledNodeCount"], 256],
            "routerIndices": [report["batch"]["sampledNodeCount"], 2],
            "routerWeights": [report["batch"]["sampledNodeCount"], 2],
        }
        assert set(item["router"]["observedExpertIndices"]).issubset(item["allowedExpertIndices"])
        assert len(item["allowedExpertMask"]) == 7
        assert len(item["outputHash"]) == 64
    assert report["reportHash"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "reportHash"}
    )
    assert _bundle_snapshot() == before


def test_bounded_russia_batch_is_deterministic_and_label_independent() -> None:
    export = _read_export(BUNDLED_MODEL_ROOT)
    russia = _load_russia(BUNDLED_MODEL_ROOT, export)

    first = _bounded_russia_batch(russia)
    second = _bounded_russia_batch(russia)

    assert first.seed_node_ids == second.seed_node_ids
    assert first.batch_hash == second.batch_hash
    assert first.maximum_node_count == SEED_NODE_COUNT * sum(
        NEIGHBOR_FANOUT**depth for depth in range(NEIGHBOR_HOPS + 1)
    )
    assert first.node_ids.size <= first.maximum_node_count
    assert first.edge_index.shape[0] == 2
    assert first.edge_index.shape[1] <= first.maximum_node_count**2


def test_checkpoint_reference_rejects_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-published-checkpoint")

    with pytest.raises(ValueError, match="do not match the export"):
        _safe_checkpoint(
            tmp_path,
            {
                "checkpointPath": "checkpoint.pt",
                "checkpointSha256": "0" * 64,
            },
        )


def test_forward_smoke_cli_emits_one_machine_json_document(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "run_checkpoint_forward_smoke",
        lambda root, *, device: {
            "schemaVersion": FORWARD_SMOKE_SCHEMA,
            "passed": True,
            "readOnly": True,
            "device": device,
            "protocolCount": 4,
            "protocols": [{"protocol": value} for value in FORWARD_SMOKE_PROTOCOLS],
        },
    )

    assert cli.main(["forward-smoke", "--root", str(tmp_path), "--device", "cpu"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["command"] == "forward-smoke"
    assert payload["readOnly"] is True
    assert tuple(item["protocol"] for item in payload["protocols"]) == FORWARD_SMOKE_PROTOCOLS


def test_cuda_forward_smoke_uses_cuda_or_fails_closed_when_unavailable() -> None:
    if torch.cuda.is_available():
        report = run_checkpoint_forward_smoke(BUNDLED_MODEL_ROOT, device="cuda")
        assert report["passed"] is True
        assert report["device"] == "cuda"
        assert all(item["device"] == "cuda" for item in report["protocols"])
    else:
        with pytest.raises(ValueError, match="CUDA was requested but is unavailable"):
            run_checkpoint_forward_smoke(BUNDLED_MODEL_ROOT, device="cuda")
