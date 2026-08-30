import random

import pytest

from socialgraph_gfm.checkpoint import (
    capture_rng_state,
    load_baseline_checkpoint,
    read_baseline_manifest,
    restore_rng_state,
    save_baseline_checkpoint,
)
from socialgraph_gfm.errors import CheckpointIntegrityError
from socialgraph_gfm.runtime import runtime_report


CPU_AVAILABLE = runtime_report("cpu")["runtimeReady"]
pytestmark = pytest.mark.skipif(
    not CPU_AVAILABLE, reason="exact CPU Torch/PyG/OGB runtime is not installed"
)


def test_baseline_checkpoint_roundtrip_is_safe_restorable_and_non_promotable(tmp_path):
    import numpy as np
    import torch

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    rng = capture_rng_state()
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()))
    restore_rng_state(rng)
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert float(torch.rand(())) == expected_torch

    manifest = save_baseline_checkpoint(
        tmp_path,
        checkpoint_id="run-best",
        run_id="run",
        epoch=4,
        track="ogb_official",
        model="mlp",
        model_state={"weight": torch.arange(4, dtype=torch.float32)},
        predictor_state={"bias": torch.ones(1)},
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=None,
        sampler_state={"epoch": 4, "seed": 20260812},
        selection_rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 1}},
        best_validation_hits50=0.42,
        best_epoch=4,
        best_model_state={"weight": torch.arange(4, dtype=torch.float32)},
        best_predictor_state={"bias": torch.ones(1)},
        selected_batch_size=4096,
        evaluations_without_improvement=0,
        history=[{"epoch": 4.0, "loss": 0.5}],
        terminal=False,
        config={"name": "baseline"},
        corpus_hash="a" * 64,
        verification_digest="b" * 64,
        rng_state=rng,
    )
    assert manifest.registrable is False
    restored_manifest = read_baseline_manifest(tmp_path / "run-best.manifest.json")
    payload = load_baseline_checkpoint(restored_manifest)
    assert payload["epoch"] == 4
    assert torch.equal(payload["model_state"]["weight"], torch.arange(4, dtype=torch.float32))


def test_baseline_checkpoint_rejects_artifact_tampering(tmp_path):
    import torch

    manifest = save_baseline_checkpoint(
        tmp_path,
        checkpoint_id="tamper",
        run_id="run",
        epoch=1,
        track="strict_edge_time",
        model="graphsage",
        model_state={"weight": torch.ones(1)},
        predictor_state={},
        optimizer_state={},
        scheduler_state=None,
        sampler_state={"epoch": 1},
        selection_rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 1}},
        best_validation_hits50=0.1,
        best_epoch=1,
        best_model_state={"weight": torch.ones(1)},
        best_predictor_state={},
        selected_batch_size=1024,
        evaluations_without_improvement=0,
        history=[{"epoch": 1.0, "loss": 0.7}],
        terminal=False,
        config={"name": "baseline"},
        corpus_hash="a" * 64,
    )
    with open(manifest.artifact_path, "ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        load_baseline_checkpoint(manifest)
