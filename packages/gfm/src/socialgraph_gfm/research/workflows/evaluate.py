"""SocialGraph-FM Research evaluation, calibration checks, and comparison claim gates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from ..routing import task_route_domain
from .common import (
    EVALUATION_SCHEMA,
    _atomic_json,
    _safe_root,
    load_research_config,
)
from .runtime import _load_trained_runtime
from .train import (
    COMPARISON_CHECKPOINT_SCHEMA,
    _binary_logit,
    _bundle_edge_index,
    _ece,
    _load_tolokers_folds,
    _role_ids,
    load_comparison_manifest,
)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("average precision requires aligned nonempty values")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("average precision requires binary labels")
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0
    grouped: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        if not math.isfinite(score):
            raise ValueError("average precision scores must be finite")
        grouped.setdefault(score, []).append(label)
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        positives = sum(group)
        true_positives += positives
        false_positives += len(group) - positives
        recall = true_positives / positive_count
        precision = true_positives / (true_positives + false_positives)
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = math.fsum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _macro_f1(labels: list[int], scores: list[float]) -> float:
    predictions = [int(score >= 0.5) for score in scores]
    values: list[float] = []
    for target in (0, 1):
        true_positive = sum(
            observed == target and predicted == target
            for observed, predicted in zip(labels, predictions, strict=True)
        )
        false_positive = sum(
            observed != target and predicted == target
            for observed, predicted in zip(labels, predictions, strict=True)
        )
        false_negative = sum(
            observed == target and predicted != target
            for observed, predicted in zip(labels, predictions, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / 2








def _scores_from_logits(logits, calibrator: Mapping[str, Any]) -> tuple[list[float], bool]:
    import torch

    binary = _binary_logit(logits).detach().to(dtype=torch.float64, device="cpu")
    adequate = bool(calibrator["adequate"])
    selected = (
        (binary + float(calibrator["bias"])) / float(calibrator["temperature"])
        if adequate
        else binary
    )
    return [float(value) for value in torch.sigmoid(selected).tolist()], adequate


def _binary_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    return {
        "macro-f1": _macro_f1(labels, scores),
        "auprc": _average_precision(labels, scores),
        "auroc": _auroc(labels, scores),
        "ece": _ece(labels, scores),
        "exampleCount": len(labels),
    }


def _node_task_evaluation(
    *,
    model,
    documents,
    adapters,
    task_id: str,
    domains: tuple[str, ...],
    calibrator: Mapping[str, Any],
    device: str,
) -> dict[str, float]:
    import torch

    labels_all: list[int] = []
    logits_all = []
    head = model.content_policy_head if task_id == CONTENT_POLICY_TASK else model.account_risk_head
    with torch.inference_mode():
        for domain in domains:
            bundle, labels, _entry = documents[domain]
            encoded = model.encode_domain(
                adapters[domain](), _bundle_edge_index(bundle, visible_only=True).to(device), domain
            )
            label_by_id = {item["entityId"]: int(item["target"]) for item in labels["targets"]}
            node_by_id = {node.id: node.index for node in bundle.nodes}
            selected = [item for item in _role_ids(bundle, "test") if item in label_by_id]
            if not selected:
                raise ValueError(f"research test partition is empty for {domain}")
            indices = torch.tensor([node_by_id[item] for item in selected], device=device)
            logits_all.append(head(encoded[indices]))
            labels_all.extend(label_by_id[item] for item in selected)
    if not labels_all:
        raise ValueError(f"research evaluation has no held-out labels for {task_id}")
    logits = torch.cat(logits_all)
    raw_scores, _ = _scores_from_logits(logits, {**calibrator, "adequate": False})
    scores, calibrated = _scores_from_logits(logits, calibrator)
    return {
        **_binary_metrics(labels_all, scores),
        "raw-ece": _ece(labels_all, raw_scores),
        "calibrated": calibrated,
    }


def _tolokers_task_evaluation(
    *, root: Path, model, documents, adapters, checkpoint, calibrators, device: str
) -> dict[str, Any]:
    import torch

    bundle, labels, _entry = documents["tolokers"]
    labels_by_id = {item["entityId"]: int(item["target"]) for item in labels["targets"]}
    folds = _load_tolokers_folds(root, documents)
    head_states = checkpoint.get("tolokersFoldHeadStates")
    if not isinstance(head_states, Mapping) or len(head_states) != 10:
        raise ValueError("checkpoint lacks all ten Tolokers split heads")
    fold_calibrators = checkpoint["tolokersFoldCalibrators"]
    with torch.inference_mode():
        encoded = model.encode_domain(
            adapters["tolokers"](),
            _bundle_edge_index(bundle, visible_only=True).to(device),
            "tolokers",
        )
    fold_rows: list[dict[str, Any]] = []
    all_labels: list[int] = []
    all_scores: list[float] = []
    for fold in folds:
        index = int(fold["fold"])
        state = head_states.get(index, head_states.get(str(index)))
        if not isinstance(state, Mapping):
            raise TypeError(f"checkpoint is missing Tolokers split {index} head")
        model.account_risk_head.load_state_dict(state, strict=True)
        model.account_risk_head.eval()
        selected = tuple(int(item) for item in fold["test"])
        if not selected:
            raise ValueError(f"Tolokers split {index} has no test nodes")
        indices = torch.tensor(selected, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model.account_risk_head(encoded[indices])
        observed = [labels_by_id[bundle.nodes[item].id] for item in selected]
        scores, calibrated = _scores_from_logits(logits, fold_calibrators[index]["calibrator"])
        metrics = _binary_metrics(observed, scores)
        fold_rows.append({"fold": index, **metrics, "calibrated": calibrated})
        all_labels.extend(observed)
        all_scores.extend(scores)
    first_state = head_states.get(0, head_states.get("0"))
    model.account_risk_head.load_state_dict(first_state, strict=True)
    model.account_risk_head.eval()
    metric_names = ("auprc", "auroc", "macro-f1", "ece")
    aggregate: dict[str, Any] = {
        name: math.fsum(float(item[name]) for item in fold_rows) / len(fold_rows)
        for name in metric_names
    }
    aggregate["exampleCount"] = sum(int(item["exampleCount"]) for item in fold_rows)
    aggregate["metricStd"] = {
        name: math.sqrt(
            math.fsum((float(item[name]) - aggregate[name]) ** 2 for item in fold_rows)
            / len(fold_rows)
        )
        for name in metric_names
    }
    pooled = _binary_metrics(all_labels, all_scores)
    return {
        **aggregate,
        "officialSplitProtocol": "10-overlapping-official-splits/1.0",
        "officialSplitCount": 10,
        "officialSplits": fold_rows,
        "calibrated": all(bool(item["calibrated"]) for item in fold_rows),
        "officialSplitInventoryHash": canonical_sha256(fold_rows),
        "pooledSecondary": pooled,
    }


def _signed_task_evaluation(
    *, model, documents, adapters, calibrator: Mapping[str, Any], device: str
) -> dict[str, float]:
    import torch

    bundle, labels, _entry = documents["wiki-rfa"]
    by_label = {item["entityId"]: item for item in labels["targets"]}
    selected = [item for item in _role_ids(bundle, "test") if item in by_label]
    if not selected:
        raise ValueError("Wiki-RfA evaluation has no held-out directed relations")
    by_id = {node.id: node.index for node in bundle.nodes}
    pairs = torch.tensor(
        [
            (by_id[by_label[item]["sourceId"]], by_id[by_label[item]["targetId"]])
            for item in selected
        ],
        device=device,
    )
    with torch.inference_mode():
        encoded = model.encode_domain(
            adapters["wiki-rfa"](),
            _bundle_edge_index(bundle, visible_only=True).to(device),
            "wiki-rfa",
        )
        logits = model.signed_edge_head(encoded, pairs)
    scores, calibrated = _scores_from_logits(logits, calibrator)
    raw_scores, _ = _scores_from_logits(logits, {**calibrator, "adequate": False})
    observed = [int(by_label[item]["target"]) for item in selected]
    negative_labels = [1 - value for value in observed]
    negative_scores = [1.0 - float(value) for value in scores]
    return {
        "negative-auprc": _average_precision(negative_labels, negative_scores),
        "macro-f1": _macro_f1(observed, [float(value) for value in scores]),
        "auroc": _auroc(observed, [float(value) for value in scores]),
        "ece": _ece(observed, [float(value) for value in scores]),
        "raw-ece": _ece(observed, raw_scores),
        "exampleCount": len(observed),
        "calibrated": calibrated,
    }


def _average_tie_rank(scores: list[float], target_position: int) -> float:
    if not scores or not 0 <= target_position < len(scores):
        raise ValueError("filtered rank requires a valid target score")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("filtered rank scores must be finite")
    target_score = scores[target_position]
    greater = sum(score > target_score for score in scores)
    equal_others = sum(score == target_score for score in scores) - 1
    return 1.0 + greater + 0.5 * equal_others


def _email_filtered_rankings(
    *,
    num_nodes: int,
    positive_pairs: tuple[tuple[int, int], ...],
    visible_train_pairs: set[tuple[int, int]],
    all_true_pairs: set[tuple[int, int]],
    gfm_score_candidates: Callable[[int, list[int]], list[float]],
    department_by_index: Mapping[int, str],
) -> dict[str, Any]:
    if num_nodes < 2 or not positive_pairs:
        raise ValueError("Email filtered ranking requires nodes and positive pairs")
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for left, right in visible_train_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    ranks: dict[str, list[float]] = {
        "gfm": [],
        "common-neighbors": [],
        "adamic-adar": [],
    }
    department_ranks: dict[str, list[float]] = {
        "same-department": [],
        "cross-department": [],
    }
    for first, second in positive_pairs:
        for anchor, target in ((first, second), (second, first)):
            candidates = [
                node
                for node in range(num_nodes)
                if node != anchor
                and (node == target or tuple(sorted((anchor, node))) not in all_true_pairs)
            ]
            try:
                target_position = candidates.index(target)
            except ValueError as error:
                raise ValueError("Email filtered candidate set omitted its target") from error
            common_scores = [
                float(len(adjacency[anchor] & adjacency[candidate]))
                for candidate in candidates
            ]
            adamic_scores = [
                math.fsum(
                    1.0 / math.log(len(adjacency[neighbor]))
                    for neighbor in adjacency[anchor] & adjacency[candidate]
                    if len(adjacency[neighbor]) > 1
                )
                for candidate in candidates
            ]
            method_scores = {
                "gfm": gfm_score_candidates(anchor, candidates),
                "common-neighbors": common_scores,
                "adamic-adar": adamic_scores,
            }
            for method, values in method_scores.items():
                if len(values) != len(candidates):
                    raise ValueError("Email filtered scorer returned an invalid width")
                ranks[method].append(_average_tie_rank(values, target_position))
            group = (
                "same-department"
                if department_by_index[anchor] == department_by_index[target]
                else "cross-department"
            )
            department_ranks[group].append(ranks["gfm"][-1])
    result: dict[str, Any] = {
        "directionPolicy": "both-endpoints/1.0",
        "tiePolicy": "average-rank/1.0",
        "candidateFilter": "exclude-all-known-true-neighbors-keep-target/1.0",
        "rankingExampleCount": len(ranks["gfm"]),
    }
    for method, values in ranks.items():
        prefix = "" if method == "gfm" else f"{method}-"
        result[f"{prefix}filtered-mrr"] = math.fsum(1.0 / rank for rank in values) / len(
            values
        )
        result[f"{prefix}hits-at-10"] = math.fsum(rank <= 10 for rank in values) / len(values)
    result["offlineDepartmentGroupMetrics"] = {
        group: {
            "filtered-mrr": math.fsum(1.0 / rank for rank in values) / len(values),
            "hits-at-10": math.fsum(rank <= 10 for rank in values) / len(values),
            "exampleCount": len(values),
        }
        for group, values in department_ranks.items()
        if values
    }
    return result


def _link_task_evaluation(
    *, model, documents, adapters, calibrator: Mapping[str, Any], device: str
) -> dict[str, Any]:
    import torch

    bundle, labels, _entry = documents["email-eu-core"]
    partition = labels["partitions"]["test"]
    rows = [*partition["positives"], *partition["negatives"]]
    if not rows:
        raise ValueError("Email-Eu-core evaluation has no held-out pairs")
    by_id = {node.id: node.index for node in bundle.nodes}
    pairs = torch.tensor(
        [(by_id[item["sourceId"]], by_id[item["targetId"]]) for item in rows],
        device=device,
    )
    with torch.inference_mode():
        encoded = model.encode_domain(
            adapters["email-eu-core"](),
            _bundle_edge_index(bundle, visible_only=True).to(device),
            task_route_domain(COLLABORATION_TASK, "email-eu-core"),
        )
        logits = model.collaboration_head(encoded, pairs)
    scores, calibrated = _scores_from_logits(logits, calibrator)
    raw_scores, _ = _scores_from_logits(logits, {**calibrator, "adequate": False})
    observed = [int(item["target"]) for item in rows]

    department_by_id = {
        item["nodeId"]: item["group"]
        for item in labels.get("offlineGroups", {}).get("department", ())
    }
    if set(department_by_id) != set(by_id):
        raise ValueError("Email-EU-core offline department groups are incomplete")
    from ...core.adapters import derive_training_selection

    visible_train_pairs = {
        tuple(
            sorted(
                (
                    by_id[bundle.edges[index].source_id],
                    by_id[bundle.edges[index].target_id],
                )
            )
        )
        for index in derive_training_selection(bundle).visible_edge_indices
    }
    adjacency: list[set[int]] = [set() for _ in bundle.nodes]
    for left, right in visible_train_pairs:
        adjacency[left].add(right)
        adjacency[right].add(left)
    common_neighbor_scores = [
        float(len(adjacency[left] & adjacency[right])) for left, right in pairs.cpu().tolist()
    ]
    adamic_scores = []
    for left, right in pairs.cpu().tolist():
        score = 0.0
        for neighbor in adjacency[left] & adjacency[right]:
            degree = len(adjacency[neighbor])
            if degree > 1:
                import math

                score += 1.0 / math.log(degree)
        adamic_scores.append(score)
    def gfm_score_candidates(anchor: int, candidates: list[int]) -> list[float]:
        values: list[float] = []
        with torch.inference_mode():
            for offset in range(0, len(candidates), 8_192):
                chunk = candidates[offset : offset + 8_192]
                candidate_pairs = torch.tensor(
                    [(anchor, node) for node in chunk], dtype=torch.long, device=device
                )
                values.extend(
                    float(value)
                    for value in model.collaboration_head(encoded, candidate_pairs).cpu().tolist()
                )
        return values

    filtered = _email_filtered_rankings(
        num_nodes=len(bundle.nodes),
        positive_pairs=tuple(
            (by_id[item["sourceId"]], by_id[item["targetId"]])
            for item in partition["positives"]
        ),
        visible_train_pairs=visible_train_pairs,
        all_true_pairs={
            tuple(sorted((by_id[edge.source_id], by_id[edge.target_id])))
            for edge in bundle.edges
        },
        gfm_score_candidates=gfm_score_candidates,
        department_by_index={
            index: department_by_id[node.id] for index, node in enumerate(bundle.nodes)
        },
    )
    return {
        **filtered,
        "auprc": _average_precision(observed, [float(value) for value in scores]),
        "auroc": _auroc(observed, [float(value) for value in scores]),
        "macro-f1": _macro_f1(observed, [float(value) for value in scores]),
        "ece": _ece(observed, [float(value) for value in scores]),
        "raw-ece": _ece(observed, raw_scores),
        "common-neighbors-auprc": _average_precision(observed, common_neighbor_scores),
        "adamic-adar-auprc": _average_precision(observed, adamic_scores),
        "exampleCount": len(observed),
        "calibrated": calibrated,
        "heuristicVisibleTopologyHash": canonical_sha256(sorted(visible_train_pairs)),
        "departmentUsedAsModelInput": False,
    }


def _load_comparison_runtime(
    *, root: Path, item: Mapping[str, Any], corpus, documents, device: str
):
    import torch

    from ...core.adapters import AdapterSchema, BundleInputAdapter
    from ...core.model import ResearchCoreGFM

    path = root / "runs/comparisons" / item["checkpointPath"]
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint.get("schemaVersion") != COMPARISON_CHECKPOINT_SCHEMA
        or checkpoint.get("corpusHash") != corpus["corpusHash"]
        or checkpoint.get("cell", {}).get("cellId") != item["cellId"]
        or checkpoint.get("variant") != item["variant"]
        or checkpoint.get("protocol", {}).get("protocolHash") != item["protocolHash"]
    ):
        raise ValueError("research comparison checkpoint binding mismatch")
    calibrator = checkpoint.get("calibrator")
    if calibrator.get("artifactHash") != canonical_sha256(
        {key: value for key, value in calibrator.items() if key != "artifactHash"}
    ):
        raise ValueError("research comparison calibrator hash mismatch")
    domains = tuple(checkpoint["domains"])
    model = ResearchCoreGFM(domains=domains).to(device)
    model.load_state_dict(checkpoint["modelState"], strict=True)
    model.eval()
    domain = item["targetDomain"]
    schema = AdapterSchema.model_validate_json(canonical_json(checkpoint["adapterSchema"]))
    adapter = BundleInputAdapter(documents[domain][0], schema=schema, mode="training").to(device)
    adapter.load_state_dict(checkpoint["adapterState"], strict=True)
    adapter.eval()
    return checkpoint, model, adapter, calibrator


def _comparison_account_evaluation(
    *, root: Path, item: Mapping[str, Any], model, adapter, documents, calibrator, device: str
) -> dict[str, Any]:
    import torch

    bundle, labels, _entry = documents["tolokers"]
    fold = _load_tolokers_folds(root, documents)[int(item["fold"])]
    selected = tuple(int(value) for value in fold["test"])
    if not selected:
        raise ValueError("Tolokers comparison split has no test nodes")
    label_by_id = {item["entityId"]: int(item["target"]) for item in labels["targets"]}
    with torch.inference_mode():
        encoded = model.encode_domain(
            adapter(), _bundle_edge_index(bundle, visible_only=True).to(device), "tolokers"
        )
        logits = model.account_risk_head(
            encoded[torch.tensor(selected, dtype=torch.long, device=device)]
        )
    observed = [label_by_id[bundle.nodes[index].id] for index in selected]
    scores, calibrated = _scores_from_logits(logits, calibrator)
    return {**_binary_metrics(observed, scores), "calibrated": calibrated}


def _comparison_claim_gate(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        CONTENT_POLICY_TASK,
        ACCOUNT_RISK_TASK,
        SIGNED_RELATION_TASK,
        COLLABORATION_TASK,
    }
    if set(aggregates) != expected:
        raise ValueError("comparison claim gate requires exactly four governance tasks")
    deltas = [float(aggregates[task]["sharedVsScratchDelta"]) for task in sorted(expected)]
    qualifying = sum(value > 0 for value in deltas)
    average_delta = math.fsum(deltas) / len(deltas)
    return {
        "claimStatus": (
            "observed_transfer_gain"
            if qualifying >= 3 and average_delta > 0
            else "not_demonstrated"
        ),
        "qualifyingTaskCount": qualifying,
        "averagePrimaryMetricDelta": average_delta,
        "requiredQualifyingTaskCount": 3,
        "requiresPositiveAverageDelta": True,
    }


def _evaluate_comparison_matrix(
    *, root: Path, corpus, documents, device: str
) -> dict[str, Any] | None:
    matrix_path = root / "runs/comparisons/matrix-manifest.json"
    if not matrix_path.is_file():
        return None
    matrix = load_comparison_manifest(root)
    observations: list[dict[str, Any]] = []
    primary_names = {
        CONTENT_POLICY_TASK: "macro-f1",
        ACCOUNT_RISK_TASK: "auprc",
        SIGNED_RELATION_TASK: "negative-auprc",
        COLLABORATION_TASK: "filtered-mrr",
    }
    for item in matrix["runs"]:
        _checkpoint, model, adapter, calibrator = _load_comparison_runtime(
            root=root,
            item=item,
            corpus=corpus,
            documents=documents,
            device=device,
        )
        task_id = item["taskId"]
        if task_id == CONTENT_POLICY_TASK:
            metrics = _node_task_evaluation(
                model=model,
                documents=documents,
                adapters={item["targetDomain"]: adapter},
                task_id=task_id,
                domains=(item["targetDomain"],),
                calibrator=calibrator,
                device=device,
            )
        elif task_id == ACCOUNT_RISK_TASK:
            metrics = _comparison_account_evaluation(
                root=root,
                item=item,
                model=model,
                adapter=adapter,
                documents=documents,
                calibrator=calibrator,
                device=device,
            )
        elif task_id == SIGNED_RELATION_TASK:
            metrics = _signed_task_evaluation(
                model=model,
                documents=documents,
                adapters={"wiki-rfa": adapter},
                calibrator=calibrator,
                device=device,
            )
        else:
            metrics = _link_task_evaluation(
                model=model,
                documents=documents,
                adapters={"email-eu-core": adapter},
                calibrator=calibrator,
                device=device,
            )
        observations.append(
            {
                "cellId": item["cellId"],
                "taskId": task_id,
                "targetDomain": item["targetDomain"],
                "fold": item["fold"],
                "variant": item["variant"],
                "protocolHash": item["protocolHash"],
                "checkpointSha256": item["checkpointSha256"],
                "primaryMetric": primary_names[task_id],
                "metrics": metrics,
                "testReadAt": "evaluate-only",
            }
        )
    aggregates: dict[str, Any] = {}
    variants = tuple(matrix["variants"])
    for task_id, primary_name in primary_names.items():
        task_payload: dict[str, Any] = {
            "primaryMetric": primary_name,
            "variants": {},
        }
        for variant in variants:
            values = [
                float(item["metrics"][primary_name])
                for item in observations
                if item["taskId"] == task_id and item["variant"] == variant
            ]
            if not values:
                raise ValueError("comparison evaluation lacks a task/variant cell")
            mean = math.fsum(values) / len(values)
            task_payload["variants"][variant] = {
                "mean": mean,
                "std": math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values)),
                "cellCount": len(values),
            }
        scratch = task_payload["variants"]["graphsage-scratch"]["mean"]
        shared = task_payload["variants"]["target-excluded-shared-gfm"]["mean"]
        single = task_payload["variants"]["single-domain-masked-pretrain"]["mean"]
        task_payload["sharedVsScratchDelta"] = shared - scratch
        task_payload["sharedVsSingleDomainDelta"] = shared - single
        aggregates[task_id] = task_payload
    claim_gate = _comparison_claim_gate(aggregates)
    payload = {
        "matrixHash": matrix["matrixHash"],
        "runCount": len(observations),
        "testRole": "evaluate-only",
        "observations": observations,
        "aggregates": aggregates,
        "claimGate": claim_gate,
    }
    payload["comparisonEvaluationHash"] = canonical_sha256(payload)
    return payload


def evaluate_research_model(research_root: str | Path, *, device: str = "cpu") -> Path:
    root = _safe_root(research_root)
    output = root / "reports/evaluation.json"
    if output.exists():
        raise FileExistsError(f"research evaluation already exists: {output}")
    training, checkpoint, corpus, documents, model, adapters = _load_trained_runtime(
        root, device=device
    )
    calibrators = checkpoint["calibrators"]
    metrics = {
        CONTENT_POLICY_TASK: _node_task_evaluation(
            model=model,
            documents=documents,
            adapters=adapters,
            task_id=CONTENT_POLICY_TASK,
            domains=tuple(item for item in sorted(documents) if item.startswith("twitch-")),
            calibrator=calibrators[CONTENT_POLICY_TASK],
            device=device,
        ),
        ACCOUNT_RISK_TASK: _tolokers_task_evaluation(
            root=root,
            model=model,
            documents=documents,
            adapters=adapters,
            checkpoint=checkpoint,
            calibrators=calibrators,
            device=device,
        ),
        SIGNED_RELATION_TASK: _signed_task_evaluation(
            model=model,
            documents=documents,
            adapters=adapters,
            calibrator=calibrators[SIGNED_RELATION_TASK],
            device=device,
        ),
        COLLABORATION_TASK: _link_task_evaluation(
            model=model,
            documents=documents,
            adapters=adapters,
            calibrator=calibrators[COLLABORATION_TASK],
            device=device,
        ),
    }
    per_task_calibration = {
        task_id: ("calibrated" if bool(metric["calibrated"]) else "ranking_only")
        for task_id, metric in metrics.items()
    }
    comparison = _evaluate_comparison_matrix(
        root=root, corpus=corpus, documents=documents, device=device
    )
    if comparison is None:
        methods = {
            "final-all-domain-shared-gfm": {"status": "evaluated"},
            "target-excluded-shared-gfm": {"status": "not-run"},
            "graphsage-scratch": {"status": "not-run"},
            "single-domain-masked-pretrain": {"status": "not-run"},
        }
        advantage_claim = {
            "claimStatus": "not_demonstrated",
            "qualifyingTaskCount": 0,
            "averagePrimaryMetricDelta": None,
            "reason": "COMPARISON_MATRIX_NOT_RUN",
        }
    else:
        methods = {
            "final-all-domain-shared-gfm": {"status": "evaluated"},
            "target-excluded-shared-gfm": {"status": "evaluated"},
            "graphsage-scratch": {"status": "evaluated"},
            "single-domain-masked-pretrain": {"status": "evaluated"},
        }
        gate = comparison["claimGate"]
        advantage_claim = {
            "claimStatus": gate["claimStatus"],
            "qualifyingTaskCount": gate["qualifyingTaskCount"],
            "averagePrimaryMetricDelta": gate["averagePrimaryMetricDelta"],
            "reason": (
                "GATE_PASSED"
                if gate["claimStatus"] == "observed_transfer_gain"
                else "THREE_OF_FOUR_AND_POSITIVE_MEAN_GATE_NOT_MET"
            ),
        }
    payload: dict[str, Any] = {
        "schemaVersion": EVALUATION_SCHEMA,
        "releaseId": RELEASE_ID,
        "seed": RESEARCH_SEED,
        "preliminary": True,
        "formalReadinessUnaffected": True,
        "researchConfigSha256": load_research_config()["configSha256"],
        "corpusHash": corpus["corpusHash"],
        "trainingHash": training["trainingHash"],
        "methods": methods,
        "metrics": metrics,
        "comparisonMatrix": comparison,
        "calibrationStatus": (
            "calibrated"
            if all(value == "calibrated" for value in per_task_calibration.values())
            else "ranking_only"
        ),
        "taskCalibrationStatus": per_task_calibration,
        "calibrationArtifactHashes": {
            task_id: calibrator["artifactHash"] for task_id, calibrator in calibrators.items()
        },
        "advantageClaim": advantage_claim,
    }
    payload["evaluationHash"] = canonical_sha256(payload)
    _atomic_json(output, payload)
    return output

COMPAT_EXPORTS = (
    '_average_precision',
    '_auroc',
    '_macro_f1',
    '_scores_from_logits',
    '_binary_metrics',
    '_node_task_evaluation',
    '_tolokers_task_evaluation',
    '_signed_task_evaluation',
    '_average_tie_rank',
    '_email_filtered_rankings',
    '_link_task_evaluation',
    '_load_comparison_runtime',
    '_comparison_account_evaluation',
    '_comparison_claim_gate',
    '_evaluate_comparison_matrix',
    'evaluate_research_model',
)

__all__ = [
    'evaluate_research_model',
]
