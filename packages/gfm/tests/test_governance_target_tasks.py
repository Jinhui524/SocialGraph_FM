from __future__ import annotations

import hashlib
import io
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pytest
import numpy as np
from pydantic import ValidationError

import socialgraph_gfm.governance.target_tasks as target_tasks_module
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.governance.adaptation import (
    AdaptationBinding,
    AdaptationComparisonV2,
    AdaptationGovernanceHandoff,
    TargetReviewPolicyV2,
)
from socialgraph_gfm.governance.cli import main
from socialgraph_gfm.governance.target_tasks import (
    LABEL_SET_SCHEMA_VERSION,
    TASK_BUNDLE_SCHEMA_VERSION,
    TargetLabelSetV2,
    generate_governance_target_tasks,
    reset_governance_target_tasks,
    verify_governance_target_tasks,
    verify_target_task_bundle,
)

LOCAL_CORPUS_ROOT = (
    Path(__file__).resolve().parents[3] / "var" / "gfm" / "global-model" / "corpus"
)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _binding() -> AdaptationBinding:
    return AdaptationBinding.model_validate(
        {
            "artifactId": "governance-artifact-" + "a" * 32,
            "datasetContentHash": _digest("dataset"),
            "graphVersionHash": _digest("graph"),
            "runId": "governance-" + "b" * 32,
            "requestHash": _digest("request"),
            "resultHash": _digest("result"),
            "runArtifactHash": _digest("run-artifact"),
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": _digest("model"),
            "modelStateHash": _digest("state"),
            "recipeHash": _digest("recipe"),
            "codeHash": _digest("code"),
            "seed": 1729,
        }
    )


def _with_hash(payload: dict[str, object], field: str) -> dict[str, object]:
    payload[field] = canonical_sha256(payload)
    return payload


def test_additive_v2_adaptation_models_validate_hashes_and_immutability() -> None:
    # Catches v2 records accepting mutation or losing immutable Global base outputs.
    binding = _binding().model_dump(mode="json", by_alias=True)
    policy_payload = _with_hash(
        {
            "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
            "binding": binding,
            "labelSetHash": _digest("labels"),
            "status": "ready",
            "selectedLambda": 0.25,
            "eligibleLabelCount": 16,
            "positiveCount": 8,
            "negativeCount": 8,
            "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
            "baseOutputsImmutable": True,
            "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
        },
        "policyHash",
    )
    policy = TargetReviewPolicyV2.model_validate(policy_payload)
    assert policy.base_outputs_immutable is True
    with pytest.raises(ValidationError):
        policy.selected_lambda = 1.0  # type: ignore[misc]

    rows = [
        {
            "nodeId": "node-0",
            "baseScore": 0.75,
            "baseRank": 1,
            "adaptedReviewPriority": 0.6,
            "adaptedRank": 2,
            "rankDelta": 1,
        }
    ]
    comparison_payload = _with_hash(
        {
            "schemaVersion": "socialgraph-fm.governance-adaptation-comparison/2.0",
            "binding": binding,
            "policyHash": policy.policy_hash,
            "total": 1,
            "baseOutputsImmutable": True,
            "rows": rows,
        },
        "comparisonHash",
    )
    comparison = AdaptationComparisonV2.model_validate(comparison_payload)
    assert comparison.rows[0].base_score == 0.75

    handoff_payload = _with_hash(
        {
            "schemaVersion": "socialgraph-fm.governance-adaptation-handoff/1.0",
            "binding": binding,
            "policyHash": policy.policy_hash,
            "comparisonHash": comparison.comparison_hash,
            "decision": "pending_governance_review",
            "baseModelMutation": False,
        },
        "handoffHash",
    )
    assert AdaptationGovernanceHandoff.model_validate(handoff_payload).base_model_mutation is False

    tampered = dict(policy_payload, policyHash="0" * 64)
    with pytest.raises(ValidationError, match="policyHash"):
        TargetReviewPolicyV2.model_validate(tampered)


@pytest.mark.parametrize(
    ("status", "selected_lambda"),
    (("ready", 0.0), ("insufficient_signal", 0.25), ("ready", 0.75)),
)
def test_v2_policy_status_and_lambda_are_one_consistent_state(
    status: str, selected_lambda: float
) -> None:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
        "binding": _binding().model_dump(mode="json", by_alias=True),
        "labelSetHash": _digest("labels"),
        "status": status,
        "selectedLambda": selected_lambda,
        "eligibleLabelCount": 16,
        "positiveCount": 8,
        "negativeCount": 8,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "baseOutputsImmutable": True,
        "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
    }
    payload["policyHash"] = canonical_sha256(payload)
    with pytest.raises(ValidationError, match="lambda|readiness|signal"):
        TargetReviewPolicyV2.model_validate(payload)


def _label_document(count: int = 16) -> dict[str, object]:
    rows = [
        {
            "nodeId": f"anonymous:{index:03d}",
            "label": "positive" if index < count // 2 else "negative",
            "structuralStratum": index % 4,
            "fusedDegree": 2 + index,
        }
        for index in range(count)
    ]
    payload: dict[str, object] = {
        "schemaVersion": LABEL_SET_SCHEMA_VERSION,
        "taskId": "target-b",
        "inferenceSha256": _digest("inference"),
        "labels": rows,
        "positiveCount": count // 2,
        "negativeCount": count // 2,
    }
    return _with_hash(payload, "labelSetHash")


def test_generic_label_contract_bounds_classes_conflicts_and_hash() -> None:
    # Catches Thailand/exact-16 coupling, class conflicts, and detached-label tampering.
    assert len(TargetLabelSetV2.model_validate(_label_document(8)).labels) == 8
    assert len(TargetLabelSetV2.model_validate(_label_document(256)).labels) == 256
    with pytest.raises(ValidationError, match="at least 8"):
        TargetLabelSetV2.model_validate(_label_document(6))
    imbalanced = _label_document(8)
    imbalanced["labels"] = [
        dict(row, label="positive")
        for row in imbalanced["labels"]  # type: ignore[index]
    ]
    imbalanced["positiveCount"] = 8
    imbalanced["negativeCount"] = 0
    imbalanced["labelSetHash"] = canonical_sha256(
        {key: value for key, value in imbalanced.items() if key != "labelSetHash"}
    )
    with pytest.raises(ValidationError, match="four labels per class"):
        TargetLabelSetV2.model_validate(imbalanced)
    conflict = _label_document()
    rows = list(conflict["labels"])  # type: ignore[arg-type]
    rows[-1] = dict(rows[0], label="negative")
    conflict["labels"] = rows
    conflict["labelSetHash"] = canonical_sha256(
        {key: value for key, value in conflict.items() if key != "labelSetHash"}
    )
    with pytest.raises(ValidationError, match="conflicting|duplicate"):
        TargetLabelSetV2.model_validate(conflict)
    with pytest.raises(ValidationError, match="labelSetHash"):
        TargetLabelSetV2.model_validate(dict(_label_document(), labelSetHash="0" * 64))


def _rewrite_zip(source: Path, destination: Path, transform) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = [(name, archive.read(name)) for name in archive.namelist()]
    entries = transform(entries)
    with zipfile.ZipFile(destination, "w", allowZip64=False) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


@pytest.fixture(scope="module")
def generated_tasks(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    corpus = LOCAL_CORPUS_ROOT
    if not corpus.is_dir():
        pytest.skip("trusted local Governance materialized corpus is unavailable")
    output = tmp_path_factory.mktemp("target-tasks")
    first = generate_governance_target_tasks(corpus, output)
    return output, first.zero_shot, first.few_shot


def test_real_generator_is_deterministic_and_meets_governance_constraints(
    generated_tasks: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches score-based/non-deterministic selection or wrong target population constraints.
    output, target_a, target_b = generated_tasks
    first_bytes = {path.name: path.read_bytes() for path in (target_a, target_b)}
    reset_governance_target_tasks(output)
    repeated = generate_governance_target_tasks(
        LOCAL_CORPUS_ROOT, output
    )
    assert repeated.zero_shot.read_bytes() == first_bytes[repeated.zero_shot.name]
    assert repeated.few_shot.read_bytes() == first_bytes[repeated.few_shot.name]

    verified = verify_governance_target_tasks(
        output, corpus_root=LOCAL_CORPUS_ROOT
    )
    assert tuple(item.task.node_count for item in verified) == (108, 108)
    assert all(180 <= item.task.fused_edge_count <= 260 for item in verified)
    assert all(
        item.task.modalities == ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
        for item in verified
    )
    assert verified[0].task.mode == "zero_shot"
    assert verified[0].task.display_name == "目标域网络 A"
    assert verified[0].labels is None
    assert verified[0].receipt.graph_population == "fold0_test"
    assert verified[1].task.mode == "few_shot"
    assert verified[1].task.display_name == "目标域网络 B"
    assert verified[1].receipt.graph_population == "full"
    assert verified[1].receipt.label_eligibility == "fold0_test"
    assert verified[1].labels is not None
    assert len(verified[1].labels.labels) == 16
    assert (verified[1].labels.positive_count, verified[1].labels.negative_count) == (8, 8)
    assert {(row.label, row.structural_stratum) for row in verified[1].labels.labels} == {
        (label, stratum) for label in ("positive", "negative") for stratum in range(4)
    }

    with zipfile.ZipFile(verified[1].path) as outer:
        label_receipt = json.loads(outer.read("label-receipt.json"))
        with zipfile.ZipFile(io.BytesIO(outer.read("inference.zip"))) as inner:
            target_ids = {
                row["node_id"]
                for row in csv.DictReader(io.StringIO(inner.read("nodes.csv").decode("utf-8")))
            }
    assert label_receipt["selectionRecipe"] == {
        "version": "fold0-test-target-class-centroid-cosine-v3",
        "stratification": "target-induced-fused-degree-node-id-quartile",
        "structuralStrata": 4,
        "labelsPerClass": 8,
        "labelsPerClassPerStratum": 2,
        "requiredSeedRecipe": "base-target-predicted-induced-quartile-stable-hash-v1",
        "featureBasis": "authenticated-target-input-text-features-768-float32",
        "representativeness": "per-class-unit-centroid-cosine-within-target-stratum",
        "similarityQuantizationDecimals": 12,
        "tieBreak": "anonymous-node-id-ascending",
        "scoreInputs": [],
    }
    corpus_root = LOCAL_CORPUS_ROOT
    uae = target_tasks_module.load_corpus_index(corpus_root, verify_manifests=True).load_country(
        "UAE", verify_hashes=True, verify_values=True
    )
    anonymous_to_source = {
        target_tasks_module._anonymous_node_id("UAE", node): node
        for node in range(uae.manifest.node_count)
    }
    target_nodes = tuple(sorted(anonymous_to_source[node_id] for node_id in target_ids))
    _, target_strata = target_tasks_module._target_graph_facts(uae, "UAE", target_nodes)
    eligible = set(map(int, np.flatnonzero(uae.split("full-fold-0").test_mask)))
    expected_labels: set[str] = set()
    for label in (1, 0):
        class_nodes = sorted(
            node for node in eligible & set(target_nodes) if int(uae.labels[node]) == label
        )
        vectors = np.asarray(uae.text_features[class_nodes], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.all(norms > 0)
        normalized = vectors / norms[:, None]
        centroid = normalized.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        similarities = {
            node: float(vector @ centroid)
            for node, vector in zip(class_nodes, normalized, strict=True)
        }
        for stratum in range(4):
            candidates = sorted(
                (node for node in class_nodes if target_strata[node] == stratum),
                key=lambda node: (
                    -round(similarities[node], 12),
                    target_tasks_module._anonymous_node_id("UAE", node),
                ),
            )
            assert len(candidates) >= 2
            expected_labels.update(
                target_tasks_module._anonymous_node_id("UAE", node) for node in candidates[:2]
            )
    assert {row.node_id for row in verified[1].labels.labels} == expected_labels

    for path, expected, artifact_name in (
        (repeated.zero_shot, 3, "匿名目标数据源 A"),
        (repeated.few_shot, 5, "匿名目标数据源 B"),
    ):
        with zipfile.ZipFile(path) as outer:
            assert len(outer.namelist()) == expected
            with zipfile.ZipFile(io.BytesIO(outer.read("inference.zip"))) as inner:
                assert json.loads(inner.read("manifest.json"))["displayName"] == artifact_name
                assert tuple(inner.namelist()) == (
                    "manifest.json",
                    "nodes.csv",
                    "relations.csv",
                    "features.npz",
                )
                assert all("label" not in name.lower() for name in inner.namelist())


def test_reset_preflights_the_complete_catalog_before_deleting_any_bytes(
    generated_tasks: tuple[Path, Path, Path],
) -> None:
    output, target_a, target_b = generated_tasks
    catalog_path = output / "governance-target-tasks.catalog.json"
    catalog_bytes = catalog_path.read_bytes()
    package_bytes = {target_a: target_a.read_bytes(), target_b: target_b.read_bytes()}
    generation_directory = target_a.parent
    unexpected = generation_directory / "operator-note.txt"
    unexpected.write_text("must block reset", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="inventory"):
            reset_governance_target_tasks(output)
        assert catalog_path.read_bytes() == catalog_bytes
        assert {path: path.read_bytes() for path in package_bytes} == package_bytes
    finally:
        unexpected.unlink(missing_ok=True)


def test_real_b_labels_are_exact_induced_graph_facts_and_target_stratified(
    generated_tasks: tuple[Path, Path, Path],
) -> None:
    """Catches emitting source-corpus degrees/strata as live target-graph facts."""
    _, _, target_b = generated_tasks
    with zipfile.ZipFile(target_b) as outer:
        labels = json.loads(outer.read("labels.json"))["labels"]
        with zipfile.ZipFile(io.BytesIO(outer.read("inference.zip"))) as inner:
            nodes = tuple(
                row["node_id"]
                for row in csv.DictReader(io.StringIO(inner.read("nodes.csv").decode("utf-8")))
            )
            relations = tuple(
                csv.DictReader(io.StringIO(inner.read("relations.csv").decode("utf-8")))
            )
    pairs = {tuple(sorted((row["source"], row["target"]))) for row in relations}
    degrees = {node: 0 for node in nodes}
    for source, target in pairs:
        degrees[source] += 1
        degrees[target] += 1
    ordered = sorted(nodes, key=lambda node: (degrees[node], node))
    strata = {node: min(3, position * 4 // len(ordered)) for position, node in enumerate(ordered)}

    assert all(row["fusedDegree"] == degrees[row["nodeId"]] for row in labels)
    assert all(row["structuralStratum"] == strata[row["nodeId"]] for row in labels)
    assert Counter((row["label"], row["structuralStratum"]) for row in labels) == Counter(
        {(label, stratum): 2 for label in ("positive", "negative") for stratum in range(4)}
    )


def test_bundle_verifier_rejects_traversal_tampering_and_wrong_inventory(
    generated_tasks: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    # Catches ZIP traversal, duplicate/unexpected members, and detached digest bypasses.
    _, target_a, target_b = generated_tasks
    traversal = tmp_path / "traversal.sgtask.zip"
    _rewrite_zip(target_a, traversal, lambda rows: rows + [("../escape", b"bad")])
    with pytest.raises(ValueError, match="path|inventory|member"):
        verify_target_task_bundle(traversal)

    tampered = tmp_path / "tampered.sgtask.zip"
    _rewrite_zip(
        target_b,
        tampered,
        lambda rows: [
            (name, payload + b"x" if name == "labels.json" else payload) for name, payload in rows
        ],
    )
    with pytest.raises(ValueError, match="digest|JSON|labels"):
        verify_target_task_bundle(tampered)

    wrong_inner = tmp_path / "wrong-inner.sgtask.zip"
    with zipfile.ZipFile(target_a) as outer:
        outer_rows = [(name, outer.read(name)) for name in outer.namelist()]
    with zipfile.ZipFile(io.BytesIO(dict(outer_rows)["inference.zip"])) as inner:
        inner_rows = [(name, inner.read(name)) for name in inner.namelist()]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as inner:
        for name, payload in inner_rows + [("labels.json", b"[]")]:
            inner.writestr(name, payload)
    _rewrite_zip(
        target_a,
        wrong_inner,
        lambda rows: [
            (name, stream.getvalue() if name == "inference.zip" else payload)
            for name, payload in rows
        ],
    )
    with pytest.raises(ValueError, match="four|inventory|digest"):
        verify_target_task_bundle(wrong_inner)


def test_task_document_schema_is_generic() -> None:
    assert TASK_BUNDLE_SCHEMA_VERSION == "socialgraph-fm.governance-target-task-bundle/1.0"


def test_interrupted_pair_publication_exposes_no_partial_root_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches publishing A before B when the second atomic replacement is interrupted.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "adaptation-inputs"
    monkeypatch.setattr(
        target_tasks_module,
        "_generate_bytes",
        lambda _corpus: (
            ("target-domain-a-zero.sgtask.zip", b"target-a"),
            ("target-domain-b-few.sgtask.zip", b"target-b"),
        ),
    )
    real_replace = target_tasks_module.os.replace
    replace_count = 0

    def interrupt_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated interruption before catalog commit")
        real_replace(source, destination)

    monkeypatch.setattr(target_tasks_module.os, "replace", interrupt_second_replace)

    with pytest.raises(OSError, match="before catalog commit"):
        generate_governance_target_tasks(corpus, output)

    assert not tuple(output.glob("*.sgtask.zip"))
    assert not (output / "governance-target-tasks.catalog.json").exists()


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }


def test_governance_catalog_generation_id_is_derived_from_exact_declarations(
    generated_tasks: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    source, target_a, target_b = generated_tasks
    forged = tmp_path / "forged-generation"
    forged_generation_id = "f" * 64
    generation = forged / ".governance-target-catalog" / forged_generation_id
    generation.mkdir(parents=True)
    for source_path in (target_a, target_b):
        shutil.copyfile(source_path, generation / source_path.name)
    source_catalog = json.loads(
        (source / "governance-target-tasks.catalog.json").read_text(encoding="utf-8")
    )
    source_catalog["generationId"] = forged_generation_id
    for target in source_catalog["targets"]:
        target["path"] = (
            f".governance-target-catalog/{forged_generation_id}/"
            f"{str(target['path']).rsplit('/', maxsplit=1)[-1]}"
        )
    source_catalog["catalogHash"] = canonical_sha256(
        {key: value for key, value in source_catalog.items() if key != "catalogHash"}
    )
    (forged / "governance-target-tasks.catalog.json").write_bytes(
        target_tasks_module._canonical_bytes(source_catalog)
    )
    operator_note = forged / "operator-note.txt"
    operator_note.write_text("preserve", encoding="utf-8")
    before = _file_snapshot(forged)

    with pytest.raises(ValueError, match="generation|identity"):
        verify_governance_target_tasks(
            forged,
            corpus_root=LOCAL_CORPUS_ROOT,
        )

    assert _file_snapshot(forged) == before


def test_generator_is_idempotent_and_atomically_upgrades_an_active_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "adaptation-inputs"
    output.mkdir()
    operator_note = output / "operator-note.txt"
    operator_note.write_text("preserve", encoding="utf-8")
    revision = {"value": b"v1"}

    def generated(_corpus: Path) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
        value = revision["value"]
        return (
            ("target-domain-a-zero.sgtask.zip", b"target-a-" + value),
            ("target-domain-b-few.sgtask.zip", b"target-b-" + value),
        )

    monkeypatch.setattr(target_tasks_module, "_generate_bytes", generated)
    first = generate_governance_target_tasks(corpus, output)
    first_catalog = (output / "governance-target-tasks.catalog.json").read_bytes()
    first_snapshot = _file_snapshot(output)
    first_mtimes = {
        path: path.stat().st_mtime_ns
        for path in (first.zero_shot, first.few_shot, output / "governance-target-tasks.catalog.json")
    }

    repeated = generate_governance_target_tasks(corpus, output)

    assert repeated == first
    assert _file_snapshot(output) == first_snapshot
    assert {path: path.stat().st_mtime_ns for path in first_mtimes} == first_mtimes

    revision["value"] = b"v2"
    upgraded = generate_governance_target_tasks(corpus, output)

    assert upgraded != first
    assert first.zero_shot.read_bytes() == b"target-a-v1"
    assert first.few_shot.read_bytes() == b"target-b-v1"
    assert upgraded.zero_shot.read_bytes() == b"target-a-v2"
    assert upgraded.few_shot.read_bytes() == b"target-b-v2"
    assert (output / "governance-target-tasks.catalog.json").read_bytes() != first_catalog
    assert target_tasks_module._governance_catalog_paths(output) == (
        upgraded.zero_shot,
        upgraded.few_shot,
    )
    assert operator_note.read_text(encoding="utf-8") == "preserve"


def test_failed_catalog_pointer_replace_preserves_the_active_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "adaptation-inputs"
    revision = {"value": b"v1"}

    def generated(_corpus: Path) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
        value = revision["value"]
        return (
            ("target-domain-a-zero.sgtask.zip", b"target-a-" + value),
            ("target-domain-b-few.sgtask.zip", b"target-b-" + value),
        )

    monkeypatch.setattr(target_tasks_module, "_generate_bytes", generated)
    active = generate_governance_target_tasks(corpus, output)
    catalog_path = output / "governance-target-tasks.catalog.json"
    active_catalog = catalog_path.read_bytes()
    active_packages = (active.zero_shot.read_bytes(), active.few_shot.read_bytes())
    revision["value"] = b"v2"
    real_replace = target_tasks_module.os.replace

    def fail_pointer_replace(source: Path, destination: Path) -> None:
        if Path(destination) == catalog_path:
            raise OSError("simulated catalog pointer interruption")
        real_replace(source, destination)

    monkeypatch.setattr(target_tasks_module.os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="pointer interruption"):
        generate_governance_target_tasks(corpus, output)

    assert catalog_path.read_bytes() == active_catalog
    assert target_tasks_module._governance_catalog_paths(output) == (
        active.zero_shot,
        active.few_shot,
    )
    assert (active.zero_shot.read_bytes(), active.few_shot.read_bytes()) == active_packages


@pytest.mark.parametrize("failure_phase", ["generate", "verify"])
def test_failed_catalog_upgrade_never_mutates_the_active_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "adaptation-inputs"
    revision = {"value": b"v1"}

    def generated(_corpus: Path) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
        value = revision["value"]
        if value == b"failed-generation":
            raise RuntimeError("simulated package generation failure")
        return (
            ("target-domain-a-zero.sgtask.zip", b"target-a-" + value),
            ("target-domain-b-few.sgtask.zip", b"target-b-" + value),
        )

    monkeypatch.setattr(target_tasks_module, "_generate_bytes", generated)
    active = generate_governance_target_tasks(corpus, output)
    active_snapshot = _file_snapshot(output)
    revision["value"] = b"failed-generation" if failure_phase == "generate" else b"v2"
    if failure_phase == "verify":
        real_verify = target_tasks_module._verify_generation_directory

        def fail_new_generation_verify(
            directory: Path,
            packages: tuple[tuple[str, bytes], tuple[str, bytes]],
        ) -> None:
            real_verify(directory, packages)
            if directory.exists() and packages[0][1] == b"target-a-v2":
                raise ValueError("simulated generation verification failure")

        monkeypatch.setattr(
            target_tasks_module,
            "_verify_generation_directory",
            fail_new_generation_verify,
        )

    with pytest.raises((RuntimeError, ValueError), match="generation failure|verification failure"):
        generate_governance_target_tasks(corpus, output)

    assert _file_snapshot(output) == active_snapshot
    assert target_tasks_module._governance_catalog_paths(output) == (
        active.zero_shot,
        active.few_shot,
    )


def test_operator_cli_generates_verifies_and_resets_real_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Catches CLI drift and reset deleting outside the two generated deliverables.
    corpus = LOCAL_CORPUS_ROOT
    if not corpus.is_dir():
        pytest.skip("trusted local Governance materialized corpus is unavailable")
    output = tmp_path / "adaptation-inputs"
    assert main(["reset-governance-target-tasks", "--output-dir", str(output)]) == 0
    capsys.readouterr()
    assert (
        main(["governance-target-tasks", "--corpus-root", str(corpus), "--output-dir", str(output)])
        == 0
    )
    assert "target-domain-a-zero.sgtask.zip" in capsys.readouterr().out
    keep = output / "operator-note.txt"
    keep.write_text("preserve", encoding="utf-8")
    keep_similar = output / ".target-operator-note.tmp"
    keep_similar.write_text("preserve", encoding="utf-8")
    assert (
        main(
            [
                "verify-governance-target-tasks",
                "--output-dir",
                str(output),
                "--corpus-root",
                str(corpus),
            ]
        )
        == 0
    )
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
    assert main(["reset-governance-target-tasks", "--output-dir", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output.resolve())
    assert keep.read_text(encoding="utf-8") == "preserve"
    assert keep_similar.read_text(encoding="utf-8") == "preserve"
    assert not tuple(output.glob("*.sgtask.zip"))
