from __future__ import annotations

import json
import gzip
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.bundle import (
    SourceProvenance,
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.datasets.parsers import ParsedGraph
from socialgraph_gfm.core.formal_preflight import (
    ExperimentLabels,
    FORMAL_CORPUS_REQUIREMENTS,
    FormalPreflightEvidence,
    load_formal_preflight,
    materialize_formal_experiment_dataset,
    publish_experiment_dataset,
    run_formal_preflight,
)
import socialgraph_gfm.core.formal_preflight as formal_preflight
from socialgraph_gfm.core.adapters import derive_training_selection
from socialgraph_gfm.core.experiment_data import bundle_from_parsed_graph
from socialgraph_gfm.core.splits import EdgeSplit, IndexSplit, SignedEdgeSplit
from socialgraph_gfm.core.datasets.recipes import DatasetRecipe


EXPECTED_REQUIREMENTS = {
    "facebook100.reed98": ("facebook100", "Reed98", "near-domain-source"),
    "facebook100.amherst41": ("facebook100", "Amherst41", "near-domain-source"),
    "facebook100.johns-hopkins55": (
        "facebook100",
        "Johns Hopkins55",
        "near-domain-source",
    ),
    "facebook100.cornell5": ("facebook100", "Cornell5", "near-domain-source"),
    "facebook100.penn94": ("facebook100", "Penn94", "offline-target"),
    "twitch.de": ("twitch-language", "DE", "cross-domain-source"),
    "twitch.en": ("twitch-language", "EN", "cross-domain-source"),
    "twitch.es": ("twitch-language", "ES", "cross-domain-source"),
    "twitch.fr": ("twitch-language", "FR", "cross-domain-source"),
    "twitch.pt": ("twitch-language", "PT", "cross-domain-source"),
    "twitch.ru": ("twitch-language", "RU", "cross-domain-source"),
    "tolokers": ("tolokers", "tolokers", "governance-target"),
    "wiki-rfa": ("wiki-rfa", "wiki-rfa", "governance-target"),
    "github-musae": ("github-musae", "github-musae", "relation-target"),
    "email-eu-core": ("email-eu-core", "email-eu-core", "relation-target"),
}


def _create_directory_link(link: Path, target: Path) -> None:
    """Create the real directory indirection used by ancestor-race probes."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"directory junction unavailable: {completed.stderr}")
        return
    link.symlink_to(target, target_is_directory=True)


def _source() -> SourceProvenance:
    return SourceProvenance(
        sourceName="literal-test-source",
        sourceUri="https://example.invalid/static",
        citation="literal fixture",
        sourceSha256="1" * 64,
    )


def _email_bundle() -> CoreGraphBundle:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "0", "index": 0}, {"id": "1", "index": 1}],
        "edges": [{"sourceId": "0", "targetId": "1", "edgeType": "email", "weight": 1.0}],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": _source().model_dump(mode="json", by_alias=True),
        "splitManifest": {
            "strategy": "spanning-forest-80-10-10",
            "assignments": [{"entityId": "edge:0:1", "role": "train"}],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _lock_recipe_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recipe_id: str,
    source_bytes: dict[str, bytes],
) -> None:
    original = formal_preflight.load_dataset_recipes()
    recipe = original[recipe_id]
    sources = tuple(
        source.model_copy(
            update={"expected_sha256": hashlib.sha256(source_bytes[source.source_id]).hexdigest()}
        )
        for source in recipe.sources
    )
    payload = recipe.model_dump(mode="python", by_alias=True)
    payload["sources"] = [source.model_dump(mode="python", by_alias=True) for source in sources]
    payload["recipeSha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "recipeSha256"}
    )
    locked = DatasetRecipe.model_validate(payload)
    catalog = {**original, recipe_id: locked}
    monkeypatch.setattr(formal_preflight, "load_dataset_recipes", lambda: catalog)


def _write_email_raw(root: Path) -> dict[str, bytes]:
    raw = root / "raw" / "email-eu-core" / "1.0.0"
    raw.mkdir(parents=True)
    sources = {
        "edges": gzip.compress(b"0 1\n1 2\n0 2\n"),
        "departments": gzip.compress(b"0 7\n1 7\n2 9\n"),
    }
    (raw / "email-Eu-core.txt.gz").write_bytes(sources["edges"])
    (raw / "email-Eu-core-department-labels.txt.gz").write_bytes(sources["departments"])
    return sources


def _rewrite_manifest_for_bundle(root: Path, manifest, bundle: CoreGraphBundle) -> None:
    bundle_path = root / manifest.bundle_relative_path
    bundle_bytes = (canonical_json(bundle) + "\n").encode("utf-8")
    bundle_path.write_bytes(bundle_bytes)
    manifest_path = root / manifest.manifest_relative_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["bundleSha256"] = hashlib.sha256(bundle_bytes).hexdigest()
    payload["graphVersionHash"] = bundle.graph_version_hash
    payload["splitManifestHash"] = canonical_sha256(
        bundle.split_manifest.model_dump(mode="python", by_alias=True)
    )
    split_path = root / payload["splitInventoryRelativePath"]
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    split_payload["folds"][0]["splitManifest"] = bundle.split_manifest.model_dump(
        mode="json", by_alias=True
    )
    split_payload["folds"][0]["splitManifestHash"] = payload["splitManifestHash"]
    split_payload["inventoryHash"] = canonical_sha256(
        {key: value for key, value in split_payload.items() if key != "inventoryHash"}
    )
    split_bytes = (canonical_json(split_payload) + "\n").encode("utf-8")
    split_path.write_bytes(split_bytes)
    payload["splitInventorySha256"] = hashlib.sha256(split_bytes).hexdigest()
    payload["splitManifestHashes"][0] = payload["splitManifestHash"]
    payload["manifestHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifestHash"}
    )
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _rewrite_manifest_for_labels(root: Path, manifest, labels: ExperimentLabels) -> None:
    labels_path = root / manifest.labels_relative_path
    labels_bytes = (canonical_json(labels) + "\n").encode("utf-8")
    labels_path.write_bytes(labels_bytes)
    manifest_path = root / manifest.manifest_relative_path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["labelsSha256"] = hashlib.sha256(labels_bytes).hexdigest()
    payload["labelNames"] = [target.name for target in labels.targets]
    payload["manifestHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifestHash"}
    )
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def test_fixed_formal_inventory_contains_every_required_graph_once() -> None:
    observed = {
        requirement.requirement_id: (
            requirement.recipe_id,
            requirement.graph_id,
            requirement.corpus_role,
        )
        for requirement in FORMAL_CORPUS_REQUIREMENTS
    }
    assert observed == EXPECTED_REQUIREMENTS
    assert len(FORMAL_CORPUS_REQUIREMENTS) == 15
    assert len({item.manifest_relative_path for item in FORMAL_CORPUS_REQUIREMENTS}) == 15

    by_id = {item.requirement_id: item for item in FORMAL_CORPUS_REQUIREMENTS}
    assert by_id["facebook100.penn94"].expected_split_policy == "official"
    assert by_id["facebook100.penn94"].required_usage_scope == "local-research-demo-only"
    assert by_id["wiki-rfa"].expected_split_policy == ("signed-pair-stratified-70-15-15")
    assert by_id["email-eu-core"].expected_split_policy == ("spanning-forest-80-10-10")


def test_empty_runtime_emits_deterministic_non_promotable_missing_evidence(
    tmp_path: Path,
) -> None:
    first = run_formal_preflight(tmp_path)
    second = run_formal_preflight(tmp_path)

    assert first == second
    assert first.formal_ready is False
    assert first.promotable is False
    assert first.evidence_hash == second.evidence_hash
    assert {item.status for item in first.observations} == {"missing"}
    assert {item.requirement_id for item in first.observations} == set(EXPECTED_REQUIREMENTS)
    assert list(tmp_path.iterdir()) == []


def test_audit_archives_and_raw_sources_never_become_formal_ready(tmp_path: Path) -> None:
    audit = tmp_path / ".inventory-audit"
    audit.mkdir()
    (audit / "twitch.zip").write_bytes(b"not-a-formal-corpus")
    (audit / "git_web_ml.zip").write_bytes(b"format-inventory-only")

    email_raw = tmp_path / "raw" / "email-eu-core" / "1.0.0"
    email_raw.mkdir(parents=True)
    (email_raw / "email-Eu-core.txt.gz").write_bytes(gzip.compress(b"0 1\n"))
    (email_raw / "email-Eu-core-department-labels.txt.gz").write_bytes(gzip.compress(b"0 1\n1 2\n"))

    evidence = run_formal_preflight(tmp_path)
    observed = {item.requirement_id: item for item in evidence.observations}

    assert {
        observed[f"twitch.{domain}"].status for domain in ("de", "en", "es", "fr", "pt", "ru")
    } == {"audit-only"}
    assert observed["github-musae"].status == "audit-only"
    assert observed["email-eu-core"].status == "raw-only"
    assert observed["email-eu-core"].graph_version_hash is None
    assert evidence.formal_ready is False
    assert evidence.promotable is False


def test_generic_dataset_publication_is_dev_only_until_source_hashes_are_locked(
    tmp_path: Path,
) -> None:
    bundle = _email_bundle()
    manifest = publish_experiment_dataset(
        runtime_root=tmp_path,
        requirement_id="email-eu-core",
        bundle=bundle,
        labels={"relation": {"edge:0:1": 1}},
        phase_eligibility="dev",
    )

    evidence = run_formal_preflight(tmp_path)
    observed = {item.requirement_id: item for item in evidence.observations}

    assert observed["email-eu-core"].status == "usage-ineligible"
    assert observed["email-eu-core"].manifest_hash == manifest.manifest_hash
    assert observed["email-eu-core"].graph_version_hash == bundle.graph_version_hash
    assert sum(item.status == "ready" for item in evidence.observations) == 0
    assert evidence.formal_ready is False
    assert evidence.promotable is False

    publication = tmp_path / manifest.manifest_relative_path
    raw = json.loads(publication.read_text(encoding="utf-8"))
    raw["phaseEligibility"] = "formal"
    raw["manifestHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifestHash"}
    )
    publication.write_text(canonical_json(raw) + "\n", encoding="utf-8")
    dev_observation = {
        item.requirement_id: item for item in run_formal_preflight(tmp_path).observations
    }["email-eu-core"]
    assert dev_observation.status == "usage-ineligible"
    assert dev_observation.reason_code == "source-hash-unlocked"

    another = tmp_path / "locked-source-required"
    another.mkdir()
    with pytest.raises(ValueError, match="dataset-specific formal materializer"):
        publish_experiment_dataset(
            runtime_root=another,
            requirement_id="email-eu-core",
            bundle=bundle,
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="formal",
        )


def test_formal_email_is_reparsed_from_locked_raw_and_fabricated_bundle_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_email_raw(tmp_path)
    _lock_recipe_sources(monkeypatch, recipe_id="email-eu-core", source_bytes=sources)
    requirement = next(
        item
        for item in formal_preflight.FORMAL_CORPUS_REQUIREMENTS
        if item.requirement_id == "email-eu-core"
    )
    source_hash = canonical_sha256(
        {
            source_id: hashlib.sha256(value).hexdigest()
            for source_id, value in sorted(sources.items())
        }
    )
    fabricated_payload = _email_bundle().model_dump(mode="python", by_alias=True)
    fabricated_payload["source"]["sourceSha256"] = source_hash
    fabricated_payload["nodes"] = [
        {"id": "fabricated-a", "index": 0},
        {"id": "fabricated-b", "index": 1},
    ]
    fabricated_payload["edges"] = [
        {
            "sourceId": "fabricated-a",
            "targetId": "fabricated-b",
            "edgeType": "fabricated",
            "weight": 1.0,
        }
    ]
    fabricated_payload["splitManifest"]["assignments"] = [
        {"entityId": "edge:fabricated-a:fabricated-b", "role": "train"}
    ]
    fabricated_payload["graphVersionHash"] = calculate_graph_version_hash(fabricated_payload)
    fabricated = CoreGraphBundle.model_validate(fabricated_payload)

    with pytest.raises(ValueError, match="dataset-specific formal materializer"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id=requirement.requirement_id,
            bundle=fabricated,
            labels={},
            phase_eligibility="formal",
        )

    manifest = materialize_formal_experiment_dataset(
        runtime_root=tmp_path, requirement_id=requirement.requirement_id
    )
    ready = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        requirement.requirement_id
    ]
    assert ready.status == "ready"
    assert ready.reason_code == "validated-formal-dataset"

    _rewrite_manifest_for_bundle(tmp_path, manifest, fabricated)
    rejected = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        requirement.requirement_id
    ]
    assert rejected.status == "invalid"
    assert rejected.reason_code == "formal-materialization-mismatch"


def test_formal_wiki_reparse_preserves_reciprocal_pair_role_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw" / "wiki-rfa" / "1.0.0"
    raw_dir.mkdir(parents=True)
    source_bytes = gzip.compress(
        (Path(__file__).parent / "fixtures" / "core_datasets" / "wiki-rfa.txt").read_bytes()
    )
    (raw_dir / "wiki-RfA.txt.gz").write_bytes(source_bytes)
    _lock_recipe_sources(monkeypatch, recipe_id="wiki-rfa", source_bytes={"wiki-rfa": source_bytes})

    manifest = materialize_formal_experiment_dataset(
        runtime_root=tmp_path, requirement_id="wiki-rfa"
    )
    bundle_path = tmp_path / manifest.bundle_relative_path
    bundle = CoreGraphBundle.model_validate_json(bundle_path.read_bytes())
    roles = {
        assignment.entity_id: assignment.role for assignment in bundle.split_manifest.assignments
    }
    assert roles["edge:A:B"] == roles["edge:B:A"]
    assert {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "wiki-rfa"
    ].status == "ready"

    forged = bundle.model_dump(mode="python", by_alias=True)
    forged["splitManifest"]["assignments"] = [
        {"entityId": "edge:A:B", "role": "train"},
        {"entityId": "edge:B:A", "role": "test"},
    ]
    forged["graphVersionHash"] = calculate_graph_version_hash(forged)
    _rewrite_manifest_for_bundle(tmp_path, manifest, CoreGraphBundle.model_validate(forged))
    rejected = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "wiki-rfa"
    ]
    assert rejected.status == "invalid"
    assert rejected.reason_code == "formal-materialization-mismatch"


def test_formal_email_labels_must_exactly_match_parser_derived_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_email_raw(tmp_path)
    _lock_recipe_sources(monkeypatch, recipe_id="email-eu-core", source_bytes=sources)
    manifest = materialize_formal_experiment_dataset(
        runtime_root=tmp_path, requirement_id="email-eu-core"
    )
    empty = ExperimentLabels.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-experiment-labels/1.0",
            "requirementId": "email-eu-core",
            "targets": [],
            "labelsHash": canonical_sha256(()),
        }
    )
    _rewrite_manifest_for_labels(tmp_path, manifest, empty)
    rejected = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "email-eu-core"
    ]
    assert rejected.status == "invalid"
    assert rejected.reason_code == "formal-materialization-mismatch"


def test_twitch_bundle_split_is_training_consumable_and_lodo_is_experiment_level() -> None:
    requirement = next(
        item
        for item in formal_preflight.FORMAL_CORPUS_REQUIREMENTS
        if item.requirement_id == "twitch.de"
    )
    assert requirement.expected_split_policy == "all-visible-training"
    assert requirement.experiment_split_policy == "leave-one-domain-out"
    parsed = ParsedGraph(
        graph_id="DE",
        directed=False,
        node_ids=("a", "b", "c"),
        edges=((0, 1), (1, 2)),
        multi_hot_features={"sharedAttributes": (("1",), ("2",), ())},
    )
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(),
        split=IndexSplit(train=(0, 1, 2), validation=(), test=()),
        excluded_feature_names=(),
        index_split_strategy="all-visible-training",
    )
    selection = derive_training_selection(bundle)
    assert selection.fit_row_ids == ("a", "b", "c")
    assert selection.visible_edge_indices == (0, 1)


def test_publication_size_gate_runs_before_final_directory_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(formal_preflight, "_MAX_LABEL_BYTES", 16)
    with pytest.raises(ValueError, match="labels artifact exceeds"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )
    assert not (tmp_path / "experiment-corpus" / "email-eu-core").exists()


def test_present_tampered_or_unsafe_manifest_is_invalid_not_missing(
    tmp_path: Path,
) -> None:
    bundle = _email_bundle()
    manifest = publish_experiment_dataset(
        runtime_root=tmp_path,
        requirement_id="email-eu-core",
        bundle=bundle,
        labels={"relation": {"edge:0:1": 1}},
        phase_eligibility="dev",
    )
    bundle_path = tmp_path / manifest.bundle_relative_path
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")

    tampered = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "email-eu-core"
    ]
    assert tampered.status == "invalid"
    assert tampered.reason_code == "artifact-validation-failed"

    manifest_path = tmp_path / manifest.manifest_relative_path
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["bundleRelativePath"] = "../escape.json"
    raw["manifestHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifestHash"}
    )
    manifest_path.write_text(canonical_json(raw) + "\n", encoding="utf-8")
    unsafe = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "email-eu-core"
    ]
    assert unsafe.status == "invalid"
    assert unsafe.reason_code == "manifest-invalid"


def test_preflight_publication_is_atomic_exact_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "formal-preflight.json"
    first = run_formal_preflight(tmp_path, publish_to=output)

    serialized = output.read_bytes()
    assert serialized.endswith(b"\n")
    assert serialized == (canonical_json(first) + "\n").encode("utf-8")
    assert load_formal_preflight(output) == first
    assert run_formal_preflight(tmp_path, publish_to=output) == first

    forged = first.model_dump(mode="json", by_alias=True)
    forged["observations"][0]["reasonCode"] = "forged"
    output.write_text(canonical_json(forged) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidenceHash"):
        load_formal_preflight(output)
    with pytest.raises(FileExistsError, match="conflicting formal preflight evidence"):
        run_formal_preflight(tmp_path, publish_to=output)

    extra = first.model_dump(mode="json", by_alias=True)
    extra["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        FormalPreflightEvidence.model_validate(extra)


def test_self_hashed_forged_evidence_cannot_override_fixed_inventory(tmp_path: Path) -> None:
    evidence = run_formal_preflight(tmp_path)
    forged = evidence.model_dump(mode="json", by_alias=True)
    forged["requirementsHash"] = "0" * 64
    forged["evidenceHash"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "evidenceHash"}
    )
    with pytest.raises(ValidationError, match="requirementsHash"):
        FormalPreflightEvidence.model_validate(forged)

    forged = evidence.model_dump(mode="json", by_alias=True)
    for observation in forged["observations"]:
        observation["status"] = "ready"
        observation["reasonCode"] = "validated-formal-dataset"
    forged["formalReady"] = True
    forged["promotable"] = True
    forged["evidenceHash"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "evidenceHash"}
    )
    with pytest.raises(ValidationError, match="semantic hashes"):
        FormalPreflightEvidence.model_validate(forged)


def test_labels_are_deeply_immutable_sorted_records(tmp_path: Path) -> None:
    manifest = publish_experiment_dataset(
        runtime_root=tmp_path,
        requirement_id="email-eu-core",
        bundle=_email_bundle(),
        labels={"relation": {"edge:0:1": 1}},
        phase_eligibility="dev",
    )
    document = ExperimentLabels.model_validate_json(
        (tmp_path / manifest.labels_relative_path).read_bytes()
    )
    assert document.targets[0].name == "relation"
    assert document.targets[0].values[0].entity_id == "edge:0:1"
    with pytest.raises(ValidationError, match="frozen"):
        document.targets[0].values[0].value = 0


def test_recipe_raw_size_limit_is_enforced_before_raw_only_status(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "facebook100" / "1.0.0"
    raw.mkdir(parents=True)
    (raw / "Reed98.mat").write_bytes(b"x" * 200_001)

    observed = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "facebook100.reed98"
    ]
    assert observed.status == "invalid"
    assert observed.reason_code == "raw-source-invalid"


def test_publishers_do_not_overwrite_targets_created_at_commit_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"

    def race_file(kind: str, target: Path) -> None:
        if kind == "evidence":
            target.write_bytes(b"racer")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", race_file)
    with pytest.raises(FileExistsError, match="conflicting formal preflight evidence"):
        run_formal_preflight(tmp_path, publish_to=output)
    assert output.read_bytes() == b"racer"

    monkeypatch.undo()
    dataset_root = tmp_path / "dataset-race"
    dataset_root.mkdir()

    def race_directory(kind: str, target: Path) -> None:
        if kind == "dataset":
            target.mkdir()
            (target / "sentinel").write_bytes(b"racer")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", race_directory)
    with pytest.raises(FileExistsError, match="conflicting experiment dataset"):
        publish_experiment_dataset(
            runtime_root=dataset_root,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )
    sentinel = dataset_root / "experiment-corpus" / "email-eu-core" / "sentinel"
    assert sentinel.read_bytes() == b"racer"


def test_post_rename_failure_never_deletes_a_replacement_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = tmp_path / "experiment-corpus" / "email-eu-core.saved"
    attack_blocked = False

    def replace_after_commit(kind: str, target: Path) -> None:
        nonlocal attack_blocked
        if kind != "dataset-post-rename":
            return
        try:
            target.rename(saved)
        except OSError as error:
            attack_blocked = True
            raise ValueError("published experiment dataset mutation blocked") from error
        target.mkdir()
        (target / "sentinel").write_bytes(b"racer")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_after_commit)
    with pytest.raises(ValueError, match="published experiment dataset"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    if attack_blocked:
        assert not saved.exists()
        assert not (tmp_path / "experiment-corpus" / "email-eu-core").exists()
    else:
        assert (saved / "dataset-manifest.json").is_file()
        assert (
            tmp_path / "experiment-corpus" / "email-eu-core" / "sentinel"
        ).read_bytes() == b"racer"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory durability contract")
def test_windows_directory_fsync_calls_flush_file_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = formal_preflight._FlushFileBuffers

    def observe(handle: int) -> int:
        calls.append(handle)
        return original(handle)

    monkeypatch.setattr(formal_preflight, "_FlushFileBuffers", observe)
    formal_preflight._fsync_directory(tmp_path)
    assert calls


def test_dataset_manifest_binds_complete_split_inventory(tmp_path: Path) -> None:
    manifest = publish_experiment_dataset(
        runtime_root=tmp_path,
        requirement_id="email-eu-core",
        bundle=_email_bundle(),
        labels={"relation": {"edge:0:1": 1}},
        phase_eligibility="dev",
    )

    assert manifest.schema_version == "socialgraph-fm.core-experiment-dataset/1.2"
    assert manifest.split_count == 1
    assert manifest.split_ids == ("primary",)
    assert manifest.split_manifest_hashes == (manifest.split_manifest_hash,)
    split_path = tmp_path / manifest.split_inventory_relative_path
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == manifest.split_inventory_sha256


def test_official_split_counts_are_fixed_by_the_requirement_catalog() -> None:
    by_id = {item.requirement_id: item for item in FORMAL_CORPUS_REQUIREMENTS}
    expected = {requirement_id: None for requirement_id in EXPECTED_REQUIREMENTS}
    expected["facebook100.penn94"] = 5
    expected["tolokers"] = 10
    assert {
        requirement_id: requirement.official_split_count
        for requirement_id, requirement in by_id.items()
    } == expected
    assert all(
        (requirement.expected_split_policy == "official")
        == (requirement.official_split_count is not None)
        for requirement in by_id.values()
    )


def test_post_link_evidence_failure_preserves_exact_idempotent_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"
    original_verify = formal_preflight._verify_owned_evidence
    first = True

    def fail_post_link_once(lease, *, max_bytes: int) -> bytes:
        nonlocal first
        if first:
            first = False
            raise OSError("simulated post-link reopen failure")
        return original_verify(lease, max_bytes=max_bytes)

    monkeypatch.setattr(formal_preflight, "_verify_owned_evidence", fail_post_link_once)
    with pytest.raises(OSError, match="post-link reopen"):
        run_formal_preflight(tmp_path, publish_to=output)
    preserved = output.read_bytes()

    monkeypatch.setattr(formal_preflight, "_verify_owned_evidence", original_verify)
    evidence = run_formal_preflight(tmp_path, publish_to=output)
    assert output.read_bytes() == preserved
    assert json.loads(preserved)["evidenceHash"] == evidence.evidence_hash


@pytest.mark.parametrize("existing_branch", ["preexisting", "commit-race"])
def test_exact_existing_evidence_is_held_through_post_flush_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_branch: str,
) -> None:
    output = tmp_path / "reports" / "preflight.json"
    expected = run_formal_preflight(tmp_path)
    expected_bytes = (canonical_json(expected) + "\n").encode("utf-8")
    if existing_branch == "preexisting":
        run_formal_preflight(tmp_path, publish_to=output)
    else:

        def publish_exact_racer(kind: str, target: Path) -> None:
            if kind == "evidence":
                target.write_bytes(expected_bytes)

        monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", publish_exact_racer)

    original_flush = formal_preflight._PublicationParentLease.flush

    def corrupt_after_flush(lease) -> None:
        original_flush(lease)
        output.write_bytes(b"CORRUPT-IDEMPOTENT-EVIDENCE")

    monkeypatch.setattr(
        formal_preflight._PublicationParentLease,
        "flush",
        corrupt_after_flush,
    )
    with pytest.raises((OSError, ValueError)):
        run_formal_preflight(tmp_path, publish_to=output)

    if os.name == "nt":
        assert output.read_bytes() == expected_bytes
    else:
        assert output.read_bytes() == b"CORRUPT-IDEMPOTENT-EVIDENCE"


def test_commit_race_exact_evidence_stays_held_through_temporary_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"
    expected = run_formal_preflight(tmp_path)
    expected_bytes = (canonical_json(expected) + "\n").encode("utf-8")

    def publish_exact_racer(kind: str, target: Path) -> None:
        if kind == "evidence":
            target.write_bytes(expected_bytes)

    original_remove = formal_preflight._remove_owned_file_path

    def corrupt_during_temporary_cleanup(*args, **kwargs):
        removed = original_remove(*args, **kwargs)
        output.write_bytes(b"CORRUPT-DURING-TEMP-CLEANUP")
        return removed

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", publish_exact_racer)
    monkeypatch.setattr(
        formal_preflight, "_remove_owned_file_path", corrupt_during_temporary_cleanup
    )
    with pytest.raises((OSError, ValueError)):
        run_formal_preflight(tmp_path, publish_to=output)
    if os.name == "nt":
        assert output.read_bytes() == expected_bytes
    else:
        assert output.read_bytes() == b"CORRUPT-DURING-TEMP-CLEANUP"


@pytest.mark.skipif(os.name == "nt", reason="POSIX visible-basename replacement contract")
@pytest.mark.parametrize("publication_branch", ["preexisting", "commit-race", "new"])
def test_exact_evidence_rejects_visible_basename_replacement_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_branch: str,
) -> None:
    output = tmp_path / "reports" / "preflight.json"
    expected = run_formal_preflight(tmp_path)
    expected_bytes = (canonical_json(expected) + "\n").encode("utf-8")
    if publication_branch == "preexisting":
        run_formal_preflight(tmp_path, publish_to=output)
    elif publication_branch == "commit-race":

        def publish_exact_racer(kind: str, target: Path) -> None:
            if kind == "evidence":
                target.write_bytes(expected_bytes)

        monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", publish_exact_racer)

    replacement = b"CORRUPT-VISIBLE-REPLACEMENT"
    saved = output.with_name(f".{output.name}.saved-exact")

    def replace_visible_target() -> None:
        output.rename(saved)
        output.write_bytes(replacement)

    if publication_branch == "preexisting":
        original_close = formal_preflight._PublisherLock.close

        def replace_after_lock_close(lock) -> None:
            original_close(lock)
            replace_visible_target()

        monkeypatch.setattr(
            formal_preflight._PublisherLock,
            "close",
            replace_after_lock_close,
        )
    else:
        original_remove = formal_preflight._remove_owned_file_path

        def replace_during_temporary_cleanup(*args, **kwargs):
            removed = original_remove(*args, **kwargs)
            replace_visible_target()
            return removed

        monkeypatch.setattr(
            formal_preflight,
            "_remove_owned_file_path",
            replace_during_temporary_cleanup,
        )

    with pytest.raises(ValueError, match="visible evidence identity changed"):
        run_formal_preflight(tmp_path, publish_to=output)
    assert output.read_bytes() == replacement
    assert saved.read_bytes() == expected_bytes


def test_posix_visible_binding_checks_basename_identity_through_parent_dirfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "reports" / "preflight.json"

    class FakeParentLease:
        parent = target.parent
        descriptor = 73

        @staticmethod
        def assert_confined() -> None:
            return None

    class ReplacementDetails:
        st_mode = 0o100600
        st_dev = 11
        st_ino = 99

    observed: dict[str, object] = {}

    def fake_stat(name, *, dir_fd, follow_symlinks):
        observed.update(
            name=name,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )
        return ReplacementDetails()

    lease = object.__new__(formal_preflight._OwnedFileLease)
    lease.target = target
    lease.identity = (11, 12)
    lease.deletable = False
    lease.parent_lease = FakeParentLease()
    lease.directory_lease = None
    lease._handle = None
    lease._descriptor = 71
    monkeypatch.setattr(formal_preflight.os, "name", "posix")
    monkeypatch.setattr(formal_preflight.os, "stat", fake_stat)

    with pytest.raises(ValueError, match="visible evidence identity changed"):
        lease.assert_visible_binding()
    assert observed == {
        "name": "preflight.json",
        "dir_fd": 73,
        "follow_symlinks": False,
    }


def test_post_link_owned_evidence_mismatch_is_removed_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"

    def corrupt_owned_link(kind: str, target: Path) -> None:
        if kind == "evidence-post-link":
            target.write_bytes(b"corrupt")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", corrupt_owned_link)
    with pytest.raises(ValueError, match="atomic evidence publication"):
        run_formal_preflight(tmp_path, publish_to=output)
    if os.name != "nt":
        assert output.read_bytes() == b"corrupt"
        with pytest.raises(FileExistsError, match="conflicting formal preflight evidence"):
            run_formal_preflight(tmp_path, publish_to=output)
        return
    assert not output.exists()

    monkeypatch.undo()
    evidence = run_formal_preflight(tmp_path, publish_to=output)
    assert json.loads(output.read_bytes())["evidenceHash"] == evidence.evidence_hash


def test_post_link_evidence_identity_change_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"

    def replace_owned_link(kind: str, target: Path) -> None:
        if kind == "evidence-post-link":
            target.unlink()
            target.write_bytes(b"racer")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_owned_link)
    with pytest.raises(ValueError, match="identity changed"):
        run_formal_preflight(tmp_path, publish_to=output)
    assert output.read_bytes() == b"racer"


@pytest.mark.skipif(os.name != "nt", reason="Windows rename publication identity")
def test_evidence_commit_never_accepts_a_replacement_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reports" / "preflight.json"
    original_rename = formal_preflight.os.rename
    saved: Path | None = None

    def replace_temporary(source: Path, target: Path) -> None:
        nonlocal saved
        source_path = Path(source)
        target_path = Path(target)
        if source_path.name.endswith(".tmp") and target_path == output:
            saved = source_path.with_name(source_path.name + ".saved")
            original_rename(source_path, saved)
            source_path.write_bytes(b"temporary-racer")
        original_rename(source_path, target_path)

    monkeypatch.setattr(formal_preflight.os, "rename", replace_temporary)
    with pytest.raises(ValueError, match="identity changed"):
        run_formal_preflight(tmp_path, publish_to=output)
    assert saved is not None and saved.is_file()
    assert output.read_bytes() == b"temporary-racer"


def test_staging_cleanup_never_removes_a_replacement_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement: Path | None = None

    def replace_staging(source: Path, _target: Path) -> None:
        nonlocal replacement
        saved = source.with_name(source.name + ".saved")
        source.rename(saved)
        source.mkdir()
        (source / "sentinel").write_bytes(b"staging-racer")
        replacement = source
        raise RuntimeError("simulated staging replacement")

    monkeypatch.setattr(formal_preflight, "_rename_directory_no_replace", replace_staging)
    with pytest.raises(RuntimeError, match="staging replacement"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )
    assert replacement is not None
    assert (replacement / "sentinel").read_bytes() == b"staging-racer"


def test_post_rename_rollback_preserves_same_name_child_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "experiment-corpus" / "email-eu-core"
    saved = tmp_path / "experiment-corpus" / "owned-bundle.saved"
    attack_blocked = False

    def replace_child(kind: str, committed: Path) -> None:
        nonlocal attack_blocked
        if kind != "dataset-post-rename":
            return
        try:
            (committed / "bundle.json").rename(saved)
        except OSError as error:
            attack_blocked = True
            raise ValueError("published experiment dataset child blocked") from error
        (committed / "bundle.json").write_bytes(b"COMPETITOR-REPLACEMENT")

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_child)
    with pytest.raises(ValueError, match="published experiment dataset"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    if attack_blocked:
        assert not target.exists()
        assert not saved.exists()
    else:
        assert (target / "bundle.json").read_bytes() == b"COMPETITOR-REPLACEMENT"
        assert saved.is_file()
        assert (target / "labels.json").is_file()


def test_staging_rollback_preserves_same_name_child_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement: Path | None = None
    saved: Path | None = None

    def replace_staged_child(source: Path, _target: Path) -> None:
        nonlocal replacement, saved
        saved = source.with_name(source.name + ".owned-bundle.saved")
        (source / "bundle.json").rename(saved)
        replacement = source / "bundle.json"
        replacement.write_bytes(b"COMPETITOR-REPLACEMENT")
        raise RuntimeError("simulated staged child replacement")

    monkeypatch.setattr(formal_preflight, "_rename_directory_no_replace", replace_staged_child)
    with pytest.raises(RuntimeError, match="staged child replacement"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    assert replacement is not None and saved is not None
    assert replacement.read_bytes() == b"COMPETITOR-REPLACEMENT"
    assert saved.is_file()


def test_evidence_publication_never_commits_through_replaced_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "reports" / "preflight.json"
    outside = tmp_path / "outside-reports"
    linked = False

    def replace_parent(kind: str, target: Path) -> None:
        nonlocal linked
        if kind != "evidence":
            return
        target.parent.rename(outside)
        _create_directory_link(target.parent, outside)
        linked = True

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_parent)
    with pytest.raises((OSError, ValueError)):
        run_formal_preflight(runtime, publish_to=output)

    assert not (outside / output.name).exists()
    if linked:
        os.rmdir(output.parent)


def test_dataset_publication_never_commits_through_replaced_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside-corpus"
    linked = False
    committed_outside = False

    def replace_parent(kind: str, target: Path) -> None:
        nonlocal linked, committed_outside
        if kind == "dataset":
            target.parent.rename(outside)
            _create_directory_link(target.parent, outside)
            linked = True
        elif kind == "dataset-post-rename":
            committed_outside = (outside / target.name).exists()

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_parent)
    with pytest.raises((OSError, ValueError)):
        publish_experiment_dataset(
            runtime_root=runtime,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    assert not (outside / "email-eu-core").exists()
    assert not committed_outside
    if linked:
        os.rmdir(runtime / "experiment-corpus")


@pytest.mark.parametrize("publisher", ["evidence", "dataset"])
def test_publisher_cleanup_never_unlinks_a_replacement_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publisher: str
) -> None:
    runtime = tmp_path / publisher
    runtime.mkdir()
    saved_lock: Path | None = None
    replacement_lock: Path | None = None
    attack_blocked = False

    def replace_lock(kind: str, target: Path) -> None:
        nonlocal saved_lock, replacement_lock, attack_blocked
        if kind != publisher:
            return
        lock = target.parent / f".{target.name}.publisher.lock"
        saved = lock.with_name(lock.name + ".saved")
        try:
            lock.rename(saved)
            lock.write_bytes(b"COMPETITOR-LOCK")
        except OSError:
            attack_blocked = True
            return
        saved_lock = saved
        replacement_lock = lock

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", replace_lock)
    if publisher == "evidence":
        run_formal_preflight(runtime, publish_to=runtime / "reports" / "preflight.json")
    else:
        publish_experiment_dataset(
            runtime_root=runtime,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    if attack_blocked:
        assert saved_lock is None and replacement_lock is None
    else:
        assert saved_lock is not None and saved_lock.is_file()
        assert replacement_lock is not None
        assert replacement_lock.read_bytes() == b"COMPETITOR-LOCK"


def test_successful_dataset_publication_holds_children_through_semantic_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = formal_preflight._observe_requirement

    def mutate_after_reload(root: Path, requirement):
        observation = original(root, requirement)
        target = root / requirement.manifest_relative_path
        target.with_name("bundle.json").write_bytes(b"CORRUPT-AFTER-RELOAD")
        return observation

    monkeypatch.setattr(formal_preflight, "_observe_requirement", mutate_after_reload)
    with pytest.raises((OSError, ValueError)):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )


def test_idempotent_dataset_replay_holds_children_through_semantic_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish_experiment_dataset(
        runtime_root=tmp_path,
        requirement_id="email-eu-core",
        bundle=_email_bundle(),
        labels={"relation": {"edge:0:1": 1}},
        phase_eligibility="dev",
    )
    original = formal_preflight._observe_requirement

    def mutate_after_reload(root: Path, requirement):
        observation = original(root, requirement)
        target = root / requirement.manifest_relative_path
        target.with_name("bundle.json").write_bytes(b"CORRUPT-IDEMPOTENT-REPLAY")
        return observation

    monkeypatch.setattr(formal_preflight, "_observe_requirement", mutate_after_reload)
    with pytest.raises(FileExistsError, match="conflicting experiment dataset"):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    monkeypatch.setattr(formal_preflight, "_observe_requirement", original)
    observed = {item.requirement_id: item for item in run_formal_preflight(tmp_path).observations}[
        "email-eu-core"
    ]
    assert observed.status == ("usage-ineligible" if os.name == "nt" else "invalid")


def test_staging_identity_is_not_recaptured_after_staged_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = formal_preflight._OwnedDirectoryLease.flush
    replaced = False

    def replace_staging_after_flush(lease) -> None:
        nonlocal replaced
        original(lease)
        path = lease.target
        if replaced or not path.name.endswith(".staging"):
            return
        replaced = True
        saved = path.with_name(path.name + ".saved")
        path.rename(saved)
        shutil.copytree(saved, path)

    monkeypatch.setattr(
        formal_preflight._OwnedDirectoryLease,
        "flush",
        replace_staging_after_flush,
    )
    with pytest.raises((OSError, ValueError)):
        publish_experiment_dataset(
            runtime_root=tmp_path,
            requirement_id="email-eu-core",
            bundle=_email_bundle(),
            labels={"relation": {"edge:0:1": 1}},
            phase_eligibility="dev",
        )

    assert replaced


@pytest.mark.skipif(os.name == "nt", reason="POSIX immutable-read contract")
def test_posix_owned_read_rejects_growth_after_the_size_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b"EXPECTED")
    lease = formal_preflight._OwnedFileLease(
        target, formal_preflight._path_identity(target), deletable=False
    )
    original_read = formal_preflight.os.read
    appended = False

    def append_during_read(descriptor: int, size: int) -> bytes:
        nonlocal appended
        payload = original_read(descriptor, size)
        if not appended and payload:
            appended = True
            with target.open("ab") as stream:
                stream.write(b"-APPENDED")
        return payload

    monkeypatch.setattr(formal_preflight.os, "read", append_during_read)
    try:
        with pytest.raises(ValueError, match="grew|changed"):
            lease.read(max_bytes=1024)
    finally:
        lease.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX fail-safe cleanup contract")
def test_posix_owned_cleanup_never_pathname_unlinks_a_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "owned.lock"
    target.write_bytes(b"OWNED")
    lease = formal_preflight._OwnedFileLease(target, formal_preflight._path_identity(target))
    saved = tmp_path / "owned.saved"
    target.rename(saved)
    target.write_bytes(b"COMPETITOR")

    assert lease.remove_owned_link() is False
    assert saved.read_bytes() == b"OWNED"
    assert target.read_bytes() == b"COMPETITOR"


def test_bundle_conversion_excludes_target_and_preserves_node_split() -> None:
    parsed = ParsedGraph(
        graph_id="tiny-node",
        directed=False,
        node_ids=("a", "b", "c"),
        edges=((0, 1), (1, 2)),
        numeric_features={"attributes": ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))},
        categorical_features={
            "gender": ("1", "2", "1"),
            "group": ("x", "y", None),
        },
        multi_hot_features={"tags": (("p",), ("q", "r"), ())},
        targets={"gender": (1, 2, 1)},
    )
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(),
        split=IndexSplit(train=(0,), validation=(1,), test=(2,)),
        excluded_feature_names={"gender"},
    )

    assert [feature.name for feature in bundle.node_features] == [
        "attributes.0",
        "attributes.1",
        "group",
        "tags",
    ]
    assert {
        assignment.entity_id: assignment.role for assignment in bundle.split_manifest.assignments
    } == {
        "a": "train",
        "b": "validation",
        "c": "test",
    }
    assert bundle.graph_version_hash == calculate_graph_version_hash(bundle)


def test_partial_official_node_split_marks_unlabeled_nodes_explicitly() -> None:
    parsed = ParsedGraph(
        graph_id="partial",
        directed=False,
        node_ids=("a", "b", "c"),
        edges=((0, 1), (1, 2)),
        categorical_features={"gender": ("1", "2", None)},
        targets={"gender": (1, 2, -1)},
    )
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(),
        split=IndexSplit(train=(0,), validation=(1,), test=()),
        excluded_feature_names={"gender"},
    )
    roles = {
        assignment.entity_id: assignment.role for assignment in bundle.split_manifest.assignments
    }
    assert roles == {"a": "train", "b": "validation", "c": "unlabeled"}
    assert derive_training_selection(bundle).fit_row_ids == ("a",)


def test_signed_split_rejects_reciprocal_pair_cross_role_leakage() -> None:
    parsed = ParsedGraph(
        graph_id="wiki",
        directed=True,
        node_ids=("a", "b"),
        signed_edges=((0, 1, 1), (1, 0, -1)),
    )
    split = SignedEdgeSplit(train=((0, 1, 1),), validation=(), test=((1, 0, -1),))
    with pytest.raises(ValueError, match="unordered user pair"):
        bundle_from_parsed_graph(
            parsed,
            source=_source(),
            split=split,
            excluded_feature_names=set(),
        )


@pytest.mark.parametrize(
    ("parsed", "split", "expected_strategy", "expected_roles"),
    [
        (
            ParsedGraph(
                graph_id="tiny-link",
                directed=False,
                node_ids=("a", "b", "c"),
                edges=((0, 1), (1, 2)),
            ),
            EdgeSplit(train=((0, 1),), validation=((1, 2),), test=()),
            "spanning-forest-80-10-10",
            {"edge:a:b": "train", "edge:b:c": "validation"},
        ),
        (
            ParsedGraph(
                graph_id="tiny-signed",
                directed=True,
                node_ids=("a", "b", "c"),
                signed_edges=((0, 1, 1), (1, 2, -1)),
            ),
            SignedEdgeSplit(train=((0, 1, 1),), validation=(), test=((1, 2, -1),)),
            "signed-pair-stratified-70-15-15",
            {"edge:a:b": "train", "edge:b:c": "test"},
        ),
    ],
)
def test_bundle_conversion_preserves_edge_role_identity(
    parsed: ParsedGraph,
    split: EdgeSplit | SignedEdgeSplit,
    expected_strategy: str,
    expected_roles: dict[str, str],
) -> None:
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(),
        split=split,
        excluded_feature_names=set(),
    )
    assert bundle.split_manifest.strategy == expected_strategy
    assert {
        assignment.entity_id: assignment.role for assignment in bundle.split_manifest.assignments
    } == expected_roles
    if parsed.signed_edges:
        assert [(edge.edge_type, edge.source_id, edge.target_id) for edge in bundle.edges] == [
            ("support", "a", "b"),
            ("oppose", "b", "c"),
        ]
