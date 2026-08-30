from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from socialgraph_gfm.governance.adaptation import (
    AdaptationBinding,
    LabelEvidence,
    TargetLabelSet,
    build_target_label_set,
    fit_target_review_policy,
)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _binding(**updates: object) -> AdaptationBinding:
    payload: dict[str, object] = {
        "artifactId": "governance-artifact-" + "a" * 32,
        "datasetContentHash": _hash("dataset"),
        "graphVersionHash": _hash("graph"),
        "runId": "governance-" + "b" * 32,
        "requestHash": _hash("request"),
        "resultHash": _hash("result"),
        "runArtifactHash": _hash("run-artifact"),
        "modelVersionId": "socialgraph-fm-global/test",
        "modelVersionHash": _hash("model"),
        "modelStateHash": _hash("checkpoint"),
        "recipeHash": _hash("recipe"),
        "codeHash": _hash("code"),
        "seed": 1729,
    }
    payload.update(updates)
    return AdaptationBinding.model_validate(payload)


def _labels(*, count: int = 8, pending: bool = False) -> list[LabelEvidence]:
    labels: list[LabelEvidence] = []
    for index in range(count):
        positive = index < count // 2
        labels.append(
            LabelEvidence.model_validate(
                {
                    "nodeId": f"node-{index}",
                    "label": "pending" if pending and index == 0 else (
                        "positive" if positive else "negative"
                    ),
                    "sourceType": "concluded_review",
                    "sourceRecordId": f"event-{index:032x}",
                    "sourceRecordHash": _hash(f"event-{index}"),
                    "reviewEventHash": _hash(f"event-{index}"),
                    "binding": _binding().model_dump(mode="json", by_alias=True),
                }
            )
        )
    return labels


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows[:7], "at least eight"),
        (lambda rows: rows[:4] + rows[4:7], "at least eight"),
        (
            lambda rows: [row.model_copy(update={"label": "positive"}) for row in rows],
            "both classes",
        ),
        (
            lambda rows: rows[:3]
            + [rows[3].model_copy(update={"label": "negative"})]
            + rows[4:9],
            "at least four",
        ),
        (
            lambda rows: rows
            + [rows[0].model_copy(update={"source_record_id": "event-" + "f" * 32})],
            "duplicate node",
        ),
        (
            lambda rows: rows
            + [
                rows[0].model_copy(
                    update={
                        "label": "negative",
                        "source_record_id": "event-" + "e" * 32,
                        "source_record_hash": _hash("disagreement"),
                        "review_event_hash": _hash("disagreement"),
                    }
                )
            ],
            "conflicting",
        ),
        (
            lambda rows: rows[:1]
            + [
                rows[1].model_copy(
                    update={"binding": _binding(runId="governance-" + "c" * 32)}
                )
            ]
            + rows[2:],
            "binding",
        ),
        (
            lambda rows: rows[:1]
            + [
                rows[1].model_copy(
                    update={"binding": _binding(modelStateHash=_hash("other-state"))}
                )
            ]
            + rows[2:],
            "binding",
        ),
        (
            lambda rows: rows[:1]
            + [
                rows[1].model_copy(
                    update={"binding": _binding(graphVersionHash=_hash("other-graph"))}
                )
            ]
            + rows[2:],
            "binding",
        ),
        (
            lambda rows: rows[:1]
            + [
                rows[1].model_copy(
                    update={"binding": _binding(resultHash=_hash("other-result"))}
                )
            ]
            + rows[2:],
            "binding",
        ),
        (
            lambda rows: rows[:1]
            + [
                rows[1].model_copy(
                    update={"binding": _binding(runArtifactHash=_hash("other-run-artifact"))}
                )
            ]
            + rows[2:],
            "binding",
        ),
    ],
)
def test_label_set_rejects_ineligible_or_unbound_labels(mutation, match: str) -> None:
    # Catches acceptance of too-small, imbalanced, duplicate, conflicting, or cross-run data.
    with pytest.raises(ValueError, match=match):
        build_target_label_set(_binding(), mutation(_labels(count=9)))


def test_label_set_rejects_pending_reviews_and_tampered_hash() -> None:
    # Catches pending reviews becoming training signal and label-set identity tampering.
    with pytest.raises(ValueError, match="pending"):
        build_target_label_set(_binding(), _labels(pending=True))
    valid = build_target_label_set(_binding(), _labels())
    payload = valid.model_dump(mode="json", by_alias=True)
    payload["labelSetHash"] = "0" * 64
    with pytest.raises(ValueError, match="labelSetHash"):
        TargetLabelSet.model_validate(payload)


def test_label_set_accepts_256_sources_and_rejects_257() -> None:
    # Catches a source inventory larger than the bounded low-resource persistence contract.
    accepted = build_target_label_set(_binding(), _labels(count=256))
    assert len(accepted.labels) == 256
    with pytest.raises(ValueError, match="at most 256"):
        build_target_label_set(_binding(), _labels(count=257))


def _fit_fixture() -> tuple[TargetLabelSet, list[str], np.ndarray, np.ndarray]:
    label_set = build_target_label_set(_binding(), _labels())
    node_ids = [f"node-{index}" for index in range(12)]
    embeddings = np.zeros((12, 256), dtype=np.float32)
    for index in range(12):
        embeddings[index, index % 4] = 1.0 if index < 4 else -1.0
        embeddings[index, 8 + index] = 0.05
    logits = np.asarray([0.1, -0.2, 0.0, 0.2, 0.3, -0.1, 0.05, -0.3, 0.7, -0.6, 0.4, -0.4])
    return label_set, node_ids, logits, embeddings


def _fit(
    label_set: TargetLabelSet,
    node_ids: list[str],
    logits: np.ndarray,
    embeddings: np.ndarray,
):
    scores = 1.0 / (1.0 + np.exp(-logits))
    order = sorted(range(len(node_ids)), key=lambda index: (-float(scores[index]), node_ids[index]))
    ranks = np.empty(len(node_ids), dtype=np.int32)
    ranks[np.asarray(order)] = np.arange(1, len(node_ids) + 1, dtype=np.int32)
    return fit_target_review_policy(
        label_set,
        node_ids,
        logits,
        embeddings,
        base_scores=scores,
        base_ranks=ranks,
    )


def test_fit_is_deterministic_uses_frozen_256d_prototypes_and_keeps_base_bytes() -> None:
    # Catches trainable/mutating adaptation, wrong embedding dimension, and unstable policy IDs.
    label_set, node_ids, logits, embeddings = _fit_fixture()
    logits_before = logits.tobytes()
    embeddings_before = embeddings.tobytes()
    binding_before = json.dumps(
        label_set.binding.model_dump(mode="json", by_alias=True), sort_keys=True
    ).encode()

    first = _fit(label_set, node_ids, logits, embeddings)
    second = _fit(label_set, node_ids, logits, embeddings)

    assert first.policy.policy_hash == second.policy.policy_hash
    assert first.policy.positive_centroid_hash == second.policy.positive_centroid_hash
    assert first.policy.negative_centroid_hash == second.policy.negative_centroid_hash
    assert first.policy.selected_lambda in {0.25, 0.5, 1.0}
    assert first.policy.status == "ready"
    assert first.policy.embedding_dimension == 256
    assert logits.tobytes() == logits_before
    assert embeddings.tobytes() == embeddings_before
    assert json.dumps(
        label_set.binding.model_dump(mode="json", by_alias=True), sort_keys=True
    ).encode() == binding_before
    assert [row.base_rank for row in first.comparison.rows] == list(range(1, 13))
    assert sorted(row.adapted_rank for row in first.comparison.rows) == list(range(1, 13))
    assert all(row.rank_delta == row.adapted_rank - row.base_rank for row in first.comparison.rows)

    with pytest.raises(ValueError, match="256"):
        _fit(label_set, node_ids, logits, embeddings[:, :255])


def test_lambda_uses_deterministic_leave_one_out_balanced_log_loss() -> None:
    # Catches in-sample prototype scoring, unbalanced loss, and non-canonical lambda search.
    label_set, node_ids, logits, embeddings = _fit_fixture()
    fitted = _fit(label_set, node_ids, logits, embeddings)
    losses = fitted.policy.validation_losses
    assert tuple(losses) == ("0", "0.25", "0.5", "1")
    assert fitted.policy.selected_lambda == min(
        (0.0, 0.25, 0.5, 1.0), key=lambda value: (losses[f"{value:g}"], value)
    )
    # Changing an unlabeled node changes run-wide z-normalization, proving bound-run fitting.
    altered = embeddings.copy()
    altered[11] *= 50
    repeated = _fit(label_set, node_ids, logits, altered)
    assert repeated.policy.validation_losses != losses


def test_comparison_echoes_persisted_calibrated_scores_and_ranks() -> None:
    # Catches reconstructing baseScore/baseRank from raw logits instead of immutable outputs.npz.
    label_set, node_ids, logits, embeddings = _fit_fixture()
    persisted_scores = np.asarray(
        [0.51, 0.93, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61],
        dtype=np.float32,
    )
    persisted_ranks = np.asarray([12, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], dtype=np.int32)
    fitted = fit_target_review_policy(
        label_set,
        node_ids,
        logits,
        embeddings,
        base_scores=persisted_scores,
        base_ranks=persisted_ranks,
    )
    by_node = {row.node_id: row for row in fitted.comparison.rows}
    for index, node_id in enumerate(node_ids):
        assert by_node[node_id].base_score == float(persisted_scores[index])
        assert by_node[node_id].base_rank == int(persisted_ranks[index])
    assert [row.node_id for row in fitted.comparison.rows] == [
        node_ids[index] for index in np.argsort(persisted_ranks)
    ]


def test_zero_lambda_winner_publishes_no_ready_policy() -> None:
    # Catches publication of an adaptation when frozen embeddings add no useful signal.
    label_set = build_target_label_set(_binding(), _labels())
    node_ids = [f"node-{index}" for index in range(8)]
    logits = np.asarray([4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0])
    embeddings = np.tile(np.arange(1, 257, dtype=np.float32), (8, 1))
    fitted = _fit(label_set, node_ids, logits, embeddings)
    assert fitted.policy.selected_lambda == 0.0
    assert fitted.policy.status == "insufficient_signal"
    assert fitted.policy.ready_policy_hash is None


def test_fit_rejects_node_inventory_and_nonfinite_inputs() -> None:
    # Catches label drift and malformed frozen run arrays.
    label_set, node_ids, logits, embeddings = _fit_fixture()
    with pytest.raises(ValueError, match="eligible node"):
        _fit(label_set, node_ids[1:], logits[1:], embeddings[1:])
    bad = embeddings.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _fit(label_set, node_ids, logits, bad)
