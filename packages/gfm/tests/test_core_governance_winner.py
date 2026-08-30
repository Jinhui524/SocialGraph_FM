from __future__ import annotations

import dataclasses

import pytest

from socialgraph_gfm.core.experiments import ExperimentProtocol, build_experiment_matrix
from socialgraph_gfm.core.governance_winner import (
    GovernanceValidationObservation,
    derive_governance_winner,
)


TASKS = (
    "github.relation-completion",
    "penn94.community-resilience",
    "tolokers.risk",
    "wiki-rfa.vote-sign",
)
METHODS = ("multi-graph-shared-gfm", "domain-aligned-gfm")


def _hash(value: int) -> str:
    return f"{value:064x}"


def _observations(
    *,
    aligned_delta: float = 0.1,
    seed_bonus: dict[int, float] | None = None,
) -> tuple[GovernanceValidationObservation, ...]:
    protocol = ExperimentProtocol.fixed()
    matrix = {
        (cell.task_id, cell.method_id, cell.seed): cell
        for cell in build_experiment_matrix(protocol)
        if cell.task_id in TASKS and cell.method_id in METHODS and cell.label_budget == "full"
    }
    bonus = seed_bonus or {}
    items: list[GovernanceValidationObservation] = []
    counter = 1
    for task_index, task_id in enumerate(TASKS):
        for method_id in METHODS:
            for seed in protocol.seeds:
                cell = matrix[(task_id, method_id, seed)]
                metric = 0.5 + task_index * 0.01 + bonus.get(seed, 0.0)
                if method_id == "domain-aligned-gfm":
                    metric += aligned_delta
                items.append(
                    GovernanceValidationObservation(
                        cell_id=cell.cell_id,
                        task_id=task_id,
                        method_id=method_id,
                        seed=seed,
                        label_budget="full",
                        validation_metric=metric,
                        record_hash=_hash(counter),
                        best_checkpoint_sha256=_hash(counter + 100),
                        head_report_hash=_hash(counter + 200),
                        validation_partition_hash=_hash(counter + 300),
                        validation_protocol_hash=_hash(counter + 400),
                        validation_data_hash=_hash(counter + 500),
                        validation_callback_hash=_hash(counter + 600),
                    )
                )
                counter += 1
    return tuple(items)


def test_winner_is_validation_only_and_input_order_independent() -> None:
    protocol = ExperimentProtocol.fixed()
    seed_bonus = {seed: float(index) / 100 for index, seed in enumerate(protocol.seeds)}
    observations = _observations(aligned_delta=0.1, seed_bonus=seed_bonus)

    forward = derive_governance_winner(protocol, observations)
    reverse = derive_governance_winner(protocol, tuple(reversed(observations)))

    assert forward == reverse
    assert forward.selected_method_id == "domain-aligned-gfm"
    assert forward.selected_seed == protocol.seeds[-1]
    assert tuple(item.task_id for item in forward.selected_runs) == TASKS
    assert all(item.seed == forward.selected_seed for item in forward.selected_runs)


def test_ties_prefer_shared_and_first_protocol_seed() -> None:
    protocol = ExperimentProtocol.fixed()

    winner = derive_governance_winner(protocol, _observations(aligned_delta=0.0))

    assert winner.selected_method_id == "multi-graph-shared-gfm"
    assert winner.selected_seed == protocol.seeds[0]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-budget", "wrong-cell"])
def test_exact_fixed_40_run_inventory_is_required(mutation: str) -> None:
    protocol = ExperimentProtocol.fixed()
    observations = list(_observations())
    if mutation == "missing":
        observations.pop()
    elif mutation == "duplicate":
        observations[-1] = observations[0]
    elif mutation == "wrong-budget":
        observations[-1] = dataclasses.replace(observations[-1], label_budget="20")
    else:
        observations[-1] = dataclasses.replace(observations[-1], cell_id=_hash(9999))

    with pytest.raises(ValueError, match="inventory"):
        derive_governance_winner(protocol, tuple(observations))


def test_seed_uses_cross_task_validation_rank_sum_not_one_test_score() -> None:
    protocol = ExperimentProtocol.fixed()
    seeds = protocol.seeds
    observations = list(_observations(aligned_delta=0.1))
    updated: list[GovernanceValidationObservation] = []
    for item in observations:
        metric = item.validation_metric
        if item.method_id == "domain-aligned-gfm":
            if item.seed == seeds[1]:
                metric += 0.2
            elif item.seed == seeds[0] and item.task_id == TASKS[0]:
                metric += 1.0
        updated.append(dataclasses.replace(item, validation_metric=metric))

    winner = derive_governance_winner(protocol, tuple(updated))

    assert winner.selected_seed == seeds[1]


def test_duplicate_checkpoint_or_report_identity_is_rejected() -> None:
    protocol = ExperimentProtocol.fixed()
    observations = list(_observations())
    observations[-1] = dataclasses.replace(
        observations[-1],
        best_checkpoint_sha256=observations[0].best_checkpoint_sha256,
    )

    with pytest.raises(ValueError, match="inventory"):
        derive_governance_winner(protocol, tuple(observations))
