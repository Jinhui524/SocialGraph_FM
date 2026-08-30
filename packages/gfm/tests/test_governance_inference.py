from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from socialgraph_gfm.governance import inference as inference_module  # noqa: E402
from socialgraph_gfm.governance.inference import OnlineInferenceOutputs  # noqa: E402


def _outputs(batch_size: int, peak: float) -> OnlineInferenceOutputs:
    return OnlineInferenceOutputs(
        logits=np.zeros(2, dtype=np.float32),
        scores=np.full(2, 0.5, dtype=np.float32),
        embeddings=np.zeros((2, 256), dtype=np.float32),
        router_indices=np.asarray([[1, 7], [1, 7]], dtype=np.int16),
        router_weights=np.full((2, 2), 0.5, dtype=np.float32),
        modality_contributions=np.full((2, 2), 0.5, dtype=np.float32),
        modality_counts=np.zeros((2, 5), dtype=np.int32),
        batch_size=batch_size,
        peak_memory_mib=peak,
        seed=7,
    )


def test_cuda_oom_and_memory_ceiling_use_128_64_32_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_infer(_data, _loaded, *, batch_size, **_kwargs):
        calls.append(batch_size)
        if batch_size == 128:
            raise RuntimeError("CUDA out of memory")
        return _outputs(batch_size, 6000.0 if batch_size == 64 else 100.0)

    monkeypatch.setattr(inference_module, "_infer_once", fake_infer)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    data = SimpleNamespace(
        artifact=SimpleNamespace(dataset_content_hash="a" * 64)
    )
    loaded = SimpleNamespace(
        model_state_hash="b" * 64,
        device=torch.device("cuda"),
    )

    result = inference_module.run_online_inference(data, loaded)

    assert calls == [128, 64, 32]
    assert result.batch_size == 32
    assert result.peak_memory_mib == 100.0


def test_inference_seed_is_stable_and_binds_both_hashes() -> None:
    first = inference_module.deterministic_inference_seed("a" * 64, "b" * 64)
    second = inference_module.deterministic_inference_seed("a" * 64, "b" * 64)
    changed_dataset = inference_module.deterministic_inference_seed("c" * 64, "b" * 64)
    changed_model = inference_module.deterministic_inference_seed("a" * 64, "d" * 64)
    defense_b = inference_module.deterministic_inference_seed(
        "2ec2c6c91b3514214f853415011b8f6066b675f3f84b3ced433939b163ed3164",
        "0fbfdec3821a9483a67e2151c296fdf9c553b0cf5dd4ac9e41223044571b777f",
    )
    assert first == second
    assert first not in {changed_dataset, changed_model}
    assert all(
        0 <= value <= 9_007_199_254_740_991
        for value in (first, changed_dataset, changed_model, defense_b)
    )


def test_json_safe_seed_derivation_is_part_of_the_hashed_runtime_recipe() -> None:
    recipe = inference_module.runtime_recipe_identity(
        model_state_hash="a" * 64,
        allowed_experts=("domain:cuba", "null"),
        use_amp=False,
        execution_environment_hash="b" * 64,
    )
    assert recipe["schemaVersion"] == "socialgraph-fm.governance-runtime-recipe/2.2"
    assert recipe["executionEnvironmentHash"] == "b" * 64
    assert recipe["inferenceSeedDerivation"] == "sha256-dataset-model-u53-v2"
    assert recipe["inferenceSeedUpperExclusive"] == 2**53


def test_execution_environment_identity_binds_wheel_and_resolved_device() -> None:
    common = {
        "torch_version": "2.12.0+cu130",
        "torch_geometric_version": "2.8.0.post1",
        "pyg_lib_version": "0.7.0+pt212cu130",
        "cuda_runtime": "13.0",
    }
    cpu = inference_module.execution_environment_identity(
        device_type="cpu",
        dtype_name="float32",
        device_capability=None,
        **common,
    )
    cuda = inference_module.execution_environment_identity(
        device_type="cuda",
        dtype_name="float16",
        device_capability=(8, 9),
        **common,
    )
    assert inference_module.canonical_sha256(cpu) != inference_module.canonical_sha256(cuda)
