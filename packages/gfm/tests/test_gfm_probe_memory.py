from __future__ import annotations

import gc
from types import SimpleNamespace
import weakref

import pytest

from socialgraph_gfm import gfm_workflow as workflow


def test_probe_batches_are_lazy_bounded_and_argument_equivalent(monkeypatch) -> None:
    domains = ("academic", "software", "community")
    streams = {domain: SimpleNamespace(domain_id=domain) for domain in domains}
    calls: list[tuple[str, int, tuple[int, int], int]] = []
    lifetime = {"live": 0, "peak": 0}

    class TrackedBatch:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            lifetime["live"] += 1
            lifetime["peak"] = max(lifetime["peak"], lifetime["live"])

        def __del__(self) -> None:
            lifetime["live"] -= 1

    def fake_core_batch(stream, *, batch_size, fanout, seed):
        calls.append((stream.domain_id, batch_size, fanout, seed))
        return TrackedBatch(stream.domain_id)

    monkeypatch.setattr(workflow, "_core_batch", fake_core_batch)
    loaders = workflow._probe_batch_loaders(
        streams,
        batch_size=2048,
        fanout=(15, 10),
        seed=20260820,
    )

    # Loader construction itself must not materialise any domain batch.
    assert calls == []
    assert lifetime == {"live": 0, "peak": 0}

    previous = None
    for domain in domains:
        iterator = iter(loaders[domain])
        current = next(iterator)
        assert current.domain == domain
        if previous is not None:
            del previous
            gc.collect()
        previous = current
        try:
            next(iterator)
        except StopIteration:
            pass
        else:  # pragma: no cover - documents the one-batch contract
            raise AssertionError("a probe domain yielded more than one batch")

    del previous
    del current
    gc.collect()

    assert calls == [
        (domain, 2048, (15, 10), 20260820) for domain in domains
    ]
    # Round-robin evaluation may construct the next batch while the caller
    # still references the previous one, but no third sibling is ever eager.
    assert lifetime["peak"] <= 2
    assert lifetime["live"] == 0


@pytest.mark.parametrize("failure_kind", ("cuda", "host"))
def test_failed_probe_releases_every_candidate_resource_before_retry(
    monkeypatch, failure_kind: str
) -> None:
    import torch

    from socialgraph_gfm.gfm import model as model_module
    from socialgraph_gfm.gfm import trainer as trainer_module

    state = {"attempt": 0}
    references: dict[int, list[weakref.ReferenceType[object]]] = {1: [], 2: []}
    closed_loaders: list[int] = []

    def remember(value: object) -> None:
        references[state["attempt"]].append(weakref.ref(value))

    class FakeModel:
        def __init__(self, _config) -> None:
            if state["attempt"] == 1:
                gc.collect()
                assert closed_loaders == [1]
                assert all(reference() is None for reference in references[1])
            state["attempt"] += 1
            remember(self)

        def parameters(self):
            # The optimizer deliberately owns the model, matching real
            # parameter ownership closely enough to expose a stale optimizer.
            return [self]

    class FakeOptimizer:
        def __init__(self, parameters, **_kwargs) -> None:
            self.parameters = list(parameters)
            remember(self)

    class FakeTrainer:
        def __init__(self, model, optimizer, _config, _device) -> None:
            self.model = model
            self.optimizer = optimizer
            remember(self)

        def train_epoch(self, _loaders) -> None:
            if state["attempt"] == 1:
                if failure_kind == "cuda":
                    raise torch.OutOfMemoryError("expected first-candidate CUDA failure")
                raise MemoryError("expected first-candidate host failure")

    class FakeLoader:
        def __init__(self) -> None:
            self.attempt = state["attempt"]
            remember(self)

        def close(self) -> None:
            closed_loaders.append(self.attempt)

    class FakeStream:
        def state_dict(self):
            return {"cursor": 0}

        def load_state_dict(self, value) -> None:
            assert value == {"cursor": 0}

    optimization = SimpleNamespace(
        candidate_batch_sizes=(2048, 1024),
        effective_batch_size=4096,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip=1.0,
        cuda_memory_limit_mib=7168,
    )
    config = SimpleNamespace(
        optimization=optimization,
        architecture=SimpleNamespace(neighbor_fanout=(15, 10)),
    )

    monkeypatch.setattr(model_module, "SocialGraphFMCore", FakeModel)
    monkeypatch.setattr(trainer_module, "CoreTrainer", FakeTrainer)
    monkeypatch.setattr(torch.optim, "AdamW", FakeOptimizer)
    monkeypatch.setattr(workflow, "_model_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        workflow,
        "_probe_batch_loaders",
        lambda *_args, **_kwargs: {"domain": FakeLoader()},
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100 * 1024**2)

    selected, peak = workflow._probe_batch_size(
        config=config,
        variant="core-base",
        streams={"domain": FakeStream()},
        device="cuda",
        seed=20260820,
    )

    gc.collect()
    assert (selected, peak) == (1024, 100.0)
    assert closed_loaders == [1, 2]
    assert all(reference() is None for reference in references[2])
