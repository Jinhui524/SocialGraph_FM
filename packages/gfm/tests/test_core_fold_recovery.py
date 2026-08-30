from __future__ import annotations

import copy

import pytest
import torch

from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.fold_recovery import (
    verify_fold_recovery_state,
    verify_run_recovery_inventory,
)
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.trainer import CoreTrainer, TrainingGraph
from socialgraph_gfm.core.training_data import ExecutionPolicy, PreparedGraph

from test_core_acceptance_real import _base_bundle


def _runtime(seed: int = 17, config: TrainingConfig | None = None):
    bundle = _base_bundle("tolokers.risk", offset=0.0)
    adapter = BundleInputAdapter(bundle, mode="training")
    node_index = {item.id: item.index for item in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    train_ids = {
        item.entity_id for item in bundle.split_manifest.assignments if item.role == "train"
    }
    for edge in bundle.edges:
        if edge.source_id in train_ids and edge.target_id in train_ids:
            pairs.extend(
                (
                    (node_index[edge.source_id], node_index[edge.target_id]),
                    (node_index[edge.target_id], node_index[edge.source_id]),
                )
            )
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    graph = PreparedGraph.from_edge_index(
        num_nodes=len(bundle.nodes), edge_index=edge_index, directed=False
    )
    config = TrainingConfig.smoke(max_steps=1) if config is None else config
    policy = ExecutionPolicy(
        full_batch_edge_threshold=config.full_batch_edge_threshold,
        node_batch_size=config.node_batch_size,
        edge_batch_size=config.edge_batch_size,
        fanout=config.fanout,
    )
    model = CoreGFM(node_classes=2)
    trainer = CoreTrainer(
        model,
        {
            "tolokers::official-00": TrainingGraph.from_bundle(
                adapter=adapter,
                graph=graph,
                execution_policy=policy,
            )
        },
        config=config,
        seed=seed,
    )
    trainer.run_steps(1)
    return bundle, config, trainer.state_dict()


def test_complete_fold_state_fresh_resumes_and_binds_seed_and_adapter() -> None:
    bundle, config, state = _runtime()

    observed = verify_fold_recovery_state(
        state,
        bundle=bundle,
        adapter_domain="tolokers::official-00",
        config=config,
        expected_seed=17,
    )

    assert observed.training_seed == 17
    assert observed.optimizer_step == 1
    assert len(observed.composite_state_hash) == 64
    assert len(observed.recovery_state_hash) == 64


def test_fold_state_fresh_recovery_rederives_the_exact_nondefault_neighbor_policy() -> None:
    config = TrainingConfig(
        preset="smoke",
        min_steps=0,
        max_steps=1,
        full_batch_edge_threshold=1,
        node_batch_size=3,
        edge_batch_size=2,
        fanout=(2, 1, 0),
    )
    bundle, _config, state = _runtime(config=config)

    observed = verify_fold_recovery_state(
        state,
        bundle=bundle,
        adapter_domain="tolokers::official-00",
        config=config,
        expected_seed=17,
    )

    assert observed.optimizer_step == 1
    assert observed.training_seed == 17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fold_state_fresh_recovers_with_its_cuda_objective_rng() -> None:
    bundle = _base_bundle("tolokers.risk", offset=0.0)
    adapter = BundleInputAdapter(bundle, mode="training")
    node_index = {item.id: item.index for item in bundle.nodes}
    train_ids = {
        item.entity_id for item in bundle.split_manifest.assignments if item.role == "train"
    }
    pairs = [
        pair
        for edge in bundle.edges
        if edge.source_id in train_ids and edge.target_id in train_ids
        for pair in (
            (node_index[edge.source_id], node_index[edge.target_id]),
            (node_index[edge.target_id], node_index[edge.source_id]),
        )
    ]
    device = torch.device("cuda")
    graph = PreparedGraph.from_edge_index(
        num_nodes=len(bundle.nodes),
        edge_index=torch.tensor(pairs, dtype=torch.long).t().contiguous(),
        directed=False,
    ).to(device)
    config = TrainingConfig.smoke(max_steps=1)
    trainer = CoreTrainer(
        CoreGFM(node_classes=2).to(device),
        {
            "tolokers::official-00": TrainingGraph.from_bundle(
                adapter=adapter,
                graph=graph,
            )
        },
        config=config,
        seed=29,
    )
    trainer.run_steps(1)

    observed = verify_fold_recovery_state(
        trainer.state_dict(),
        bundle=bundle,
        adapter_domain="tolokers::official-00",
        config=config,
        expected_seed=29,
    )

    assert next(observed.model.parameters()).device.type == "cuda"
    assert observed.optimizer_step == 1


@pytest.mark.parametrize("mutation", ["missing-optimizer", "seed", "adapter"])
def test_incomplete_or_repackaged_fold_recovery_state_is_rejected(mutation: str) -> None:
    bundle, config, original = _runtime()
    state = copy.deepcopy(original)
    if mutation == "missing-optimizer":
        del state["optimizer"]
    elif mutation == "seed":
        state["trainingSeed"] = 99
    else:
        first = next(iter(state["adapters"]["tolokers::official-00"]))
        del state["adapters"]["tolokers::official-00"][first]

    with pytest.raises(ValueError):
        verify_fold_recovery_state(
            state,
            bundle=bundle,
            adapter_domain="tolokers::official-00",
            config=config,
            expected_seed=17,
        )


def test_cell_run_recovery_inventory_binds_full_state_without_a_caller_hash() -> None:
    _bundle, config, state = _runtime()
    state["experimentCellId"] = "a" * 64

    observed = verify_run_recovery_inventory(
        state,
        config=config,
        expected_seed=17,
        expected_cell_id="a" * 64,
    )

    assert observed.optimizer_step == 1
    assert observed.training_seed == 17
    assert len(observed.composite_state_hash) == 64
    assert len(observed.recovery_state_hash) == 64


def test_cell_run_recovery_inventory_rejects_rehashed_incomplete_state() -> None:
    _bundle, config, state = _runtime()
    state["experimentCellId"] = "a" * 64
    del state["optimizer"]

    with pytest.raises(ValueError, match="complete trainer recovery inventory"):
        verify_run_recovery_inventory(
            state,
            config=config,
            expected_seed=17,
            expected_cell_id="a" * 64,
        )
