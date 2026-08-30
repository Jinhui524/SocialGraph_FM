from __future__ import annotations

import numpy as np
import pytest

from socialgraph_gfm.baseline.types import CorpusArrays, RunSpec
from socialgraph_gfm.baseline_workflow import _run_one, load_baseline_config
from socialgraph_gfm.checkpoint import load_baseline_checkpoint, read_baseline_manifest
from socialgraph_gfm.registry import LocalRegistry
from socialgraph_gfm.runtime import RuntimeLayout, runtime_report


CPU_AVAILABLE = runtime_report("cpu")["runtimeReady"]
pytestmark = pytest.mark.skipif(
    not CPU_AVAILABLE, reason="exact CPU Torch/PyG/OGB runtime is not installed"
)


def _tiny_corpus() -> CorpusArrays:
    return CorpusArrays.from_mapping(
        {
            "x": np.arange(32, dtype=np.float32).reshape(8, 4) / 32,
            "edge_index": np.asarray(
                [
                    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
                ]
            ),
            "edge_timestamp": np.asarray(
                [2015, 2015, 2016, 2016, 2017, 2017, 2017, 2017, 2016, 2016, 2015, 2015]
            ),
            "variant_train_positive": np.asarray(
                [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
            ),
            "variant_validation_positive": np.asarray([[0, 1], [3, 4]]),
            "variant_validation_negative": np.asarray([[0, 0, 1, 2], [4, 5, 6, 7]]),
            "variant_test_positive": np.asarray([[0, 1, 2], [4, 5, 7]]),
            "variant_test_negative": np.asarray([[0, 1, 2, 3], [6, 7, 5, 7]]),
        },
        corpus_hash="a" * 64,
    )


def test_run_one_persists_safe_resumable_non_promotable_checkpoint(tmp_path):
    config, payload, config_hash = load_baseline_config()
    layout = RuntimeLayout(tmp_path)
    layout.prepare()
    registry = LocalRegistry(layout.registry / "registry.sqlite3")
    spec = RunSpec("dev-exp", "dev-exp-dev-ogb_official-mlp-20260811", "dev", "ogb_official", "mlp", 20260811)

    result = _run_one(
        spec,
        layout=layout,
        corpus=_tiny_corpus(),
        config=config,
        config_payload=payload,
        config_hash=config_hash,
        code_hash="b" * 64,
        environment_hash="c" * 64,
        device="cpu",
        registry=registry,
    )

    assert result["status"] == "succeeded"
    assert registry.baseline_counts() == {
        "baseline_runs": 1,
        "baseline_checkpoints": 2,
        "baseline_evaluations": 1,
        "baseline_acceptances": 0,
    }
    latest_path = (
        layout.runs
        / spec.run_id
        / "checkpoints"
        / f"{spec.run_id}-latest.manifest.json"
    )
    manifest = read_baseline_manifest(latest_path)
    checkpoint = load_baseline_checkpoint(manifest)
    assert manifest.registrable is False
    assert checkpoint["terminal"] is True
    assert checkpoint["best_epoch"] >= 1
    assert checkpoint["selection_rng_state"]["bit_generator"] == "PCG64"
