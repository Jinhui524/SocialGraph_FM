from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.governance import adaptation
from socialgraph_gfm.governance.adaptation import AdaptationBinding
from socialgraph_gfm.governance.target_tasks import (
    LABEL_SET_SCHEMA_VERSION,
    TASK_BUNDLE_SCHEMA_VERSION,
    TARGET_RECEIPT_SCHEMA_VERSION,
    TargetDomainReceipt,
    TargetLabelSetV2,
    TargetTaskDocument,
    generate_governance_target_tasks,
    verify_governance_target_tasks,
    verify_target_task_bundle,
)

LOCAL_CORPUS_ROOT = (
    Path(__file__).resolve().parents[3] / "var" / "gfm" / "global-model" / "corpus"
)


def _sha(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return stream.getvalue()


def _commit_mutated_governance_catalog(output: Path) -> None:
    names = (
        "target-domain-a-zero.sgtask.zip",
        "target-domain-b-few.sgtask.zip",
    )
    values = tuple((output / name).read_bytes() for name in names)
    declarations = [
        {"role": role, "name": name, "sha256": _sha(value), "bytes": len(value)}
        for role, name, value in zip(("zero_shot", "few_shot"), names, values, strict=True)
    ]
    generation_id = canonical_sha256({"packages": declarations})
    generation = output / ".governance-target-catalog" / generation_id
    generation.mkdir(parents=True)
    targets: list[dict[str, object]] = []
    for declaration, value in zip(declarations, values, strict=True):
        name = str(declaration["name"])
        (generation / name).write_bytes(value)
        targets.append(
            {
                "role": declaration["role"],
                "path": f".governance-target-catalog/{generation_id}/{name}",
                "sha256": declaration["sha256"],
                "bytes": declaration["bytes"],
            }
        )
    catalog: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-catalog/1.0",
        "generationId": generation_id,
        "targets": targets,
    }
    catalog["catalogHash"] = canonical_sha256(catalog)
    (output / "governance-target-tasks.catalog.json").write_bytes(_json_bytes(catalog))


def _rebind_zero_shot_inner(
    source: Path,
    destination: Path,
    mutation,
) -> None:
    """Mutate an inner member and rebind every outer/inner digest around it."""

    with zipfile.ZipFile(source) as outer:
        outer_entries = {name: outer.read(name) for name in outer.namelist()}
    with zipfile.ZipFile(io.BytesIO(outer_entries["inference.zip"])) as inner:
        inner_entries = {name: inner.read(name) for name in inner.namelist()}
    mutation(inner_entries)
    manifest = json.loads(inner_entries["manifest.json"])
    for name in ("nodes.csv", "relations.csv", "features.npz"):
        manifest["files"][name] = {
            "sha256": _sha(inner_entries[name]),
            "bytes": len(inner_entries[name]),
        }
    inner_entries["manifest.json"] = _json_bytes(manifest)
    _, node_rows = _csv_rows(inner_entries["nodes.csv"])
    node_ids = {row["node_id"] for row in node_rows}
    _, relation_rows = _csv_rows(inner_entries["relations.csv"])
    fused_edges = {
        tuple(sorted((row["source"], row["target"]))) for row in relation_rows
    }
    inference = _zip_bytes([(name, inner_entries[name]) for name in (
        "manifest.json", "nodes.csv", "relations.csv", "features.npz"
    )])
    receipt = json.loads(outer_entries["target-receipt.json"])
    receipt["inferenceSha256"] = _sha(inference)
    receipt["nodeSetSha256"] = canonical_sha256(sorted(node_ids))
    receipt["fusedEdgeCount"] = len(fused_edges)
    receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receiptHash"}
    )
    receipt_bytes = _json_bytes(receipt)
    task = json.loads(outer_entries["task.json"])
    task["fusedEdgeCount"] = len(fused_edges)
    task["inference"] = {
        "name": "inference.zip", "sha256": _sha(inference), "bytes": len(inference)
    }
    task["targetReceipt"] = {
        "name": "target-receipt.json",
        "sha256": _sha(receipt_bytes),
        "bytes": len(receipt_bytes),
    }
    destination.write_bytes(
        _zip_bytes(
            [
                ("task.json", _json_bytes(task)),
                ("inference.zip", inference),
                ("target-receipt.json", receipt_bytes),
            ]
        )
    )


def _csv_rows(value: bytes) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(value.decode("utf-8")))
    return list(reader.fieldnames or ()), list(reader)


def _csv_payload(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _mutate_duplicate_node(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["nodes.csv"])
    replaced = rows[1]["node_id"]
    rows[1]["node_id"] = rows[0]["node_id"]
    entries["nodes.csv"] = _csv_payload(fields, rows)
    relation_fields, relations = _csv_rows(entries["relations.csv"])
    for row in relations:
        if row["source"] == replaced:
            row["source"] = rows[0]["node_id"]
        if row["target"] == replaced:
            row["target"] = rows[0]["node_id"]
    entries["relations.csv"] = _csv_payload(relation_fields, relations)

    def transform(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ids = arrays["node_ids"].copy()
        ids[1] = ids[0]
        return {**arrays, "node_ids": ids}

    _rewrite_features(entries, transform)


def _mutate_dangling(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["relations.csv"])
    rows[0]["source"] = "absent-node"
    entries["relations.csv"] = _csv_payload(fields, rows)


def _mutate_self_loop(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["relations.csv"])
    rows[0]["target"] = rows[0]["source"]
    entries["relations.csv"] = _csv_payload(fields, rows)


def _mutate_unknown_modality(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["relations.csv"])
    rows[0]["modality"] = "browserScore"
    entries["relations.csv"] = _csv_payload(fields, rows)


def _mutate_unobserved_modality(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["relations.csv"])
    rows = [row for row in rows if row["modality"] != "fastRT"]
    entries["relations.csv"] = _csv_payload(fields, rows)
    manifest = json.loads(entries["manifest.json"])
    manifest["relationRowCount"] = len(rows)
    entries["manifest.json"] = _json_bytes(manifest)


def _rewrite_features(entries: dict[str, bytes], transform) -> None:
    with np.load(io.BytesIO(entries["features.npz"]), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays = transform(arrays)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    entries["features.npz"] = stream.getvalue()


def _mutate_feature_width(entries: dict[str, bytes]) -> None:
    _rewrite_features(
        entries,
        lambda arrays: {**arrays, "text_features": arrays["text_features"][:, :767]},
    )


def _mutate_feature_alignment(entries: dict[str, bytes]) -> None:
    def transform(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        changed = arrays["node_ids"].copy()
        changed[[0, 1]] = changed[[1, 0]]
        return {**arrays, "node_ids": changed}

    _rewrite_features(entries, transform)


def _mutate_extra_feature_array(entries: dict[str, bytes]) -> None:
    _rewrite_features(
        entries,
        lambda arrays: {**arrays, "labels": np.zeros(len(arrays["node_ids"]), dtype=np.uint8)},
    )


def _mutate_relation_weight(entries: dict[str, bytes]) -> None:
    fields, rows = _csv_rows(entries["relations.csv"])
    rows[0]["weight"] = format(float(rows[0]["weight"]) + 0.125, ".17g")
    entries["relations.csv"] = _csv_payload(fields, rows)


@pytest.fixture(scope="module")
def review_generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    corpus = LOCAL_CORPUS_ROOT
    if not corpus.is_dir():
        pytest.skip("trusted local Governance materialized corpus is unavailable")
    output = tmp_path_factory.mktemp("target-task-review")
    generated = generate_governance_target_tasks(corpus, output)
    return output, generated.zero_shot, generated.few_shot


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_mutate_duplicate_node, "duplicate"),
        (_mutate_dangling, "dangling"),
        (_mutate_self_loop, "self-loop"),
        (_mutate_unknown_modality, "unknown modality"),
        (_mutate_unobserved_modality, "modalities"),
        (_mutate_feature_width, "768"),
        (_mutate_feature_alignment, "align"),
        (_mutate_extra_feature_array, "array inventory|node_ids and text_features|safe NumPy"),
    ],
)
def test_verifier_semantically_inspects_rebound_inner_payload(
    review_generated: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    # Catches accepting semantically invalid bytes when all declared hashes are rebound.
    _, target_a, _ = review_generated
    tampered = tmp_path / f"{mutation.__name__}.sgtask.zip"
    _rebind_zero_shot_inner(target_a, tampered, mutation)
    with pytest.raises(ValueError, match=match):
        verify_target_task_bundle(tampered)


def _generic_task_payload() -> dict[str, Any]:
    return {
        "schemaVersion": TASK_BUNDLE_SCHEMA_VERSION,
        "taskId": "pharma-network-2028",
        "displayName": "Pharma network review",
        "mode": "zero_shot",
        "nodeCount": 37,
        "fusedEdgeCount": 42,
        "modalities": ["coURL"],
        "inference": {"name": "inference.zip", "sha256": _sha("inference"), "bytes": 10},
        "targetReceipt": {
            "name": "target-receipt.json", "sha256": _sha("receipt"), "bytes": 10
        },
    }


def test_public_target_task_models_accept_non_governance_domains_and_sizes() -> None:
    # Catches governance A/B/108/Cuba/UAE literals leaking into reusable public contracts.
    task = TargetTaskDocument.model_validate(_generic_task_payload())
    assert (task.task_id, task.node_count, task.modalities) == (
        "pharma-network-2028", 37, ("coURL",)
    )
    receipt_payload: dict[str, Any] = {
        "schemaVersion": TARGET_RECEIPT_SCHEMA_VERSION,
        "taskId": task.task_id,
        "countryId": "Brazil",
        "sourceContentHash": _sha("source"),
        "sourceManifestSha256": _sha("manifest"),
        "graphPopulation": "authorized-quarter-3",
        "graphPopulationMaskSha256": _sha("mask"),
        "labelEligibility": "none",
        "labelEligibilityMaskSha256": None,
        "inferenceSha256": _sha("inference"),
        "nodeSetSha256": _sha("nodes"),
        "nodeCount": 37,
        "fusedEdgeCount": 42,
        "modalities": ["coURL"],
        "connected": False,
        "selectionRecipe": {"version": "operator-v3", "scoreInputs": []},
    }
    receipt_payload["receiptHash"] = canonical_sha256(receipt_payload)
    receipt = TargetDomainReceipt.model_validate(receipt_payload)
    assert (receipt.country_id, receipt.graph_population) == (
        "Brazil", "authorized-quarter-3"
    )
    labels = _label_set(8, task_id=task.task_id)
    assert TargetLabelSetV2.model_validate(labels).task_id == task.task_id


def _label_set(count: int, *, task_id: str = "target-b") -> dict[str, Any]:
    rows = [
        {
            "nodeId": f"node-{index}",
            "label": "positive" if index < count // 2 else "negative",
            "structuralStratum": index % 4,
            "fusedDegree": index + 1,
        }
        for index in range(count)
    ]
    payload: dict[str, Any] = {
        "schemaVersion": LABEL_SET_SCHEMA_VERSION,
        "taskId": task_id,
        "inferenceSha256": _sha("inference"),
        "labels": rows,
        "positiveCount": count // 2,
        "negativeCount": count // 2,
    }
    payload["labelSetHash"] = canonical_sha256(payload)
    return payload


def _binding() -> AdaptationBinding:
    return AdaptationBinding.model_validate(
        {
            "artifactId": "governance-artifact-" + "a" * 32,
            "datasetContentHash": _sha("dataset"),
            "graphVersionHash": _sha("graph"),
            "runId": "governance-" + "b" * 32,
            "requestHash": _sha("request"),
            "resultHash": _sha("result"),
            "runArtifactHash": _sha("run-artifact"),
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": _sha("model"),
            "modelStateHash": _sha("state"),
            "recipeHash": _sha("recipe"),
            "codeHash": _sha("code"),
            "seed": 1729,
        }
    )


@pytest.mark.parametrize("count", [8, 16, 256])
def test_v2_label_sets_fit_frozen_policy_deterministically(count: int) -> None:
    # Catches v2 labels remaining a dead-end contract or mutating persisted Global outputs.
    label_set = TargetLabelSetV2.model_validate(_label_set(count))
    node_ids = [f"node-{index}" for index in range(count + 4)]
    logits = np.linspace(-0.4, 0.4, len(node_ids), dtype=np.float64)
    scores = np.linspace(0.2, 0.8, len(node_ids), dtype=np.float32)
    ranks = np.arange(1, len(node_ids) + 1, dtype=np.int32)
    embeddings = np.zeros((len(node_ids), 256), dtype=np.float32)
    for index in range(len(node_ids)):
        embeddings[index, index % 128] = 1.0 if index < count // 2 else -1.0
        embeddings[index, 128 + index % 128] = 0.1
    before = (logits.tobytes(), scores.tobytes(), ranks.tobytes(), embeddings.tobytes())
    fit = getattr(adaptation, "fit_target_review_policy_v2")
    first = fit(
        label_set, _binding(), node_ids, logits, embeddings,
        base_scores=scores, base_ranks=ranks,
    )
    second = fit(
        label_set, _binding(), node_ids, logits, embeddings,
        base_scores=scores, base_ranks=ranks,
    )
    assert first.policy.schema_version == "socialgraph-fm.governance-target-review-policy/2.0"
    assert first.comparison.schema_version == "socialgraph-fm.governance-adaptation-comparison/2.0"
    assert first.policy.policy_hash == second.policy.policy_hash
    assert first.comparison.comparison_hash == second.comparison.comparison_hash
    assert first.policy.binding == _binding()
    assert first.policy.label_set_hash == label_set.label_set_hash
    assert [row.base_score for row in first.comparison.rows] == list(map(float, scores))
    assert [row.base_rank for row in first.comparison.rows] == list(map(int, ranks))
    assert before == (logits.tobytes(), scores.tobytes(), ranks.tobytes(), embeddings.tobytes())


def _rebind_few_shot_labels(source: Path, destination: Path, count: int) -> None:
    with zipfile.ZipFile(source) as outer:
        entries = {name: outer.read(name) for name in outer.namelist()}
    labels = json.loads(entries["labels.json"])
    positives = [row for row in labels["labels"] if row["label"] == "positive"][: count // 2]
    negatives = [row for row in labels["labels"] if row["label"] == "negative"][: count // 2]
    labels["labels"] = positives + negatives
    labels["positiveCount"] = len(positives)
    labels["negativeCount"] = len(negatives)
    labels["labelSetHash"] = canonical_sha256(
        {key: value for key, value in labels.items() if key != "labelSetHash"}
    )
    labels_bytes = _json_bytes(labels)
    label_receipt = json.loads(entries["label-receipt.json"])
    label_receipt["labelsSha256"] = _sha(labels_bytes)
    label_receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in label_receipt.items() if key != "receiptHash"}
    )
    label_receipt_bytes = _json_bytes(label_receipt)
    task = json.loads(entries["task.json"])
    task["labels"] = {"name": "labels.json", "sha256": _sha(labels_bytes), "bytes": len(labels_bytes)}
    task["labelReceipt"] = {
        "name": "label-receipt.json", "sha256": _sha(label_receipt_bytes),
        "bytes": len(label_receipt_bytes),
    }
    entries["task.json"] = _json_bytes(task)
    entries["labels.json"] = labels_bytes
    entries["label-receipt.json"] = label_receipt_bytes
    destination.write_bytes(_zip_bytes([(name, entries[name]) for name in (
        "task.json", "inference.zip", "target-receipt.json", "labels.json", "label-receipt.json"
    )]))


def test_governance_verifier_rejects_swapped_or_duplicated_roles(
    review_generated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches treating an arbitrary pair of individually valid tasks as governance A/B.
    _, target_a, target_b = review_generated
    output = tmp_path / "swapped"
    output.mkdir()
    shutil.copyfile(target_b, output / target_a.name)
    shutil.copyfile(target_b, output / target_b.name)
    _commit_mutated_governance_catalog(output)
    with pytest.raises(ValueError, match="target-a|role|zero_shot|Cuba"):
        verify_governance_target_tasks(
            output,
            corpus_root=LOCAL_CORPUS_ROOT,
        )


def test_governance_verifier_requires_exact_label_inventory_after_rebinding(
    review_generated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches accepting the generic 8-label minimum for the exact governance B role.
    _, target_a, target_b = review_generated
    output = tmp_path / "eight-labels"
    output.mkdir()
    shutil.copyfile(target_a, output / target_a.name)
    _rebind_few_shot_labels(target_b, output / target_b.name, 8)
    assert len(verify_target_task_bundle(output / target_b.name).labels.labels) == 8  # type: ignore[union-attr]
    _commit_mutated_governance_catalog(output)
    with pytest.raises(ValueError, match="16|8/8|stratum"):
        verify_governance_target_tasks(
            output,
            corpus_root=LOCAL_CORPUS_ROOT,
        )


def test_governance_verifier_recomputes_fold0_eligibility_from_trusted_corpus(
    review_generated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches trusting a rebound eligibility declaration without reading the bound NPY mask.
    _, target_a, target_b = review_generated
    output = tmp_path / "eligibility-rebound"
    output.mkdir()
    shutil.copyfile(target_a, output / target_a.name)
    with zipfile.ZipFile(target_b) as outer:
        entries = {name: outer.read(name) for name in outer.namelist()}
    label_receipt = json.loads(entries["label-receipt.json"])
    label_nodes = {row["nodeId"] for row in json.loads(entries["labels.json"])["labels"]}
    removable = next(
        node for node in label_receipt["eligibleNodeIds"] if node not in label_nodes
    )
    label_receipt["eligibleNodeIds"].remove(removable)
    label_receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in label_receipt.items() if key != "receiptHash"}
    )
    label_receipt_bytes = _json_bytes(label_receipt)
    task = json.loads(entries["task.json"])
    task["labelReceipt"] = {
        "name": "label-receipt.json",
        "sha256": _sha(label_receipt_bytes),
        "bytes": len(label_receipt_bytes),
    }
    entries["task.json"] = _json_bytes(task)
    entries["label-receipt.json"] = label_receipt_bytes
    (output / target_b.name).write_bytes(
        _zip_bytes([(name, entries[name]) for name in (
            "task.json", "inference.zip", "target-receipt.json", "labels.json", "label-receipt.json"
        )])
    )
    assert verify_target_task_bundle(output / target_b.name).labels is not None
    _commit_mutated_governance_catalog(output)
    with pytest.raises(ValueError, match="eligib|fold-0|source|target-b"):
        verify_governance_target_tasks(
            output,
            corpus_root=LOCAL_CORPUS_ROOT,
        )


def _rebind_label_semantics(source: Path, destination: Path, mutation: str) -> None:
    with zipfile.ZipFile(source) as outer:
        entries = {name: outer.read(name) for name in outer.namelist()}
    labels = json.loads(entries["labels.json"])
    rows = labels["labels"]
    if mutation == "truth":
        left = next(row for row in rows if row["label"] == "positive")
        right = next(
            row
            for row in rows
            if row["label"] == "negative"
            and row["structuralStratum"] == left["structuralStratum"]
        )
        left["label"], right["label"] = right["label"], left["label"]
    else:
        left = next(row for row in rows if row["label"] == "positive")
        right = next(
            row
            for row in rows
            if row["label"] == left["label"]
            and row["structuralStratum"] != left["structuralStratum"]
        )
        left["structuralStratum"], right["structuralStratum"] = (
            right["structuralStratum"],
            left["structuralStratum"],
        )
    labels["labelSetHash"] = canonical_sha256(
        {key: value for key, value in labels.items() if key != "labelSetHash"}
    )
    labels_bytes = _json_bytes(labels)
    label_receipt = json.loads(entries["label-receipt.json"])
    label_receipt["labelsSha256"] = _sha(labels_bytes)
    label_receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in label_receipt.items() if key != "receiptHash"}
    )
    label_receipt_bytes = _json_bytes(label_receipt)
    task = json.loads(entries["task.json"])
    task["labels"] = {
        "name": "labels.json",
        "sha256": _sha(labels_bytes),
        "bytes": len(labels_bytes),
    }
    task["labelReceipt"] = {
        "name": "label-receipt.json",
        "sha256": _sha(label_receipt_bytes),
        "bytes": len(label_receipt_bytes),
    }
    entries["task.json"] = _json_bytes(task)
    entries["labels.json"] = labels_bytes
    entries["label-receipt.json"] = label_receipt_bytes
    destination.write_bytes(
        _zip_bytes(
            [
                (name, entries[name])
                for name in (
                    "task.json",
                    "inference.zip",
                    "target-receipt.json",
                    "labels.json",
                    "label-receipt.json",
                )
            ]
        )
    )


def test_governance_catalog_rejects_rebound_a_relation_substitution(
    review_generated: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches authenticating only A's declared shape/counts instead of exact corpus-derived bytes.
    _, target_a, target_b = review_generated
    output = tmp_path / "a-relation-substitution"
    output.mkdir()
    _rebind_zero_shot_inner(
        target_a, output / target_a.name, _mutate_relation_weight
    )
    shutil.copyfile(target_b, output / target_b.name)
    assert verify_target_task_bundle(output / target_a.name).task.task_id == "target-a"
    _commit_mutated_governance_catalog(output)
    with pytest.raises(ValueError, match="catalog|expected|target-a"):
        verify_governance_target_tasks(
            output,
            corpus_root=LOCAL_CORPUS_ROOT,
        )


@pytest.mark.parametrize("mutation", ["truth", "stratum"])
def test_governance_catalog_rejects_rebound_b_label_semantics(
    review_generated: tuple[Path, Path, Path], tmp_path: Path, mutation: str
) -> None:
    # Catches trusting B's balanced quotas without matching exact corpus truth/strata.
    _, target_a, target_b = review_generated
    output = tmp_path / f"b-label-{mutation}"
    output.mkdir()
    shutil.copyfile(target_a, output / target_a.name)
    _rebind_label_semantics(target_b, output / target_b.name, mutation)
    if mutation == "stratum":
        with pytest.raises(ValueError, match="degree|stratum|target graph"):
            verify_target_task_bundle(output / target_b.name)
        return
    assert verify_target_task_bundle(output / target_b.name).labels is not None
    _commit_mutated_governance_catalog(output)
    with pytest.raises(ValueError, match="catalog|expected|target-b"):
        verify_governance_target_tasks(
            output,
            corpus_root=LOCAL_CORPUS_ROOT,
        )


def test_v2_fitter_rejects_invalid_duck_type_before_frozen_arrays() -> None:
    # Catches bypassing concrete TargetLabelSetV2 hash/count validation via attributes.
    valid = TargetLabelSetV2.model_validate(_label_set(8))

    class InvalidDuck:
        label_set_hash = "not-a-sha256"
        positive_count = valid.positive_count
        negative_count = valid.negative_count
        labels = valid.labels

    with pytest.raises(ValidationError, match="TargetLabelSetV2|labelSetHash|sha256"):
        adaptation.fit_target_review_policy_v2(
            InvalidDuck(),  # type: ignore[arg-type]
            _binding(),
            (),
            np.empty(0, dtype=np.float64),
            np.empty((0, 256), dtype=np.float32),
            base_scores=np.empty(0, dtype=np.float32),
            base_ranks=np.empty(0, dtype=np.int32),
        )
