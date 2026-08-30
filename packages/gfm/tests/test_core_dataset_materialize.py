from __future__ import annotations

import hashlib
import json

from socialgraph_gfm.core.bundle import (
    calculate_graph_version_hash,
    load_core_graph_bundle_json,
)
import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.datasets.materialize import (
    materialize_email_eu_core,
    materialize_email_from_files,
    validate_email_materialization,
)


def test_email_materialization_is_atomic_hash_bound_and_excludes_offline_labels(tmp_path) -> None:
    edges = tmp_path / "email-Eu-core.txt"
    edges.write_text("0 1\n1 2\n2 3\n3 0\n0 2\n1 1\n", encoding="utf-8")
    departments = tmp_path / "email-Eu-core-department-labels.txt"
    departments.write_text("0 7\n1 7\n2 9\n3 9\n", encoding="utf-8")
    source_hashes = {
        "edges": hashlib.sha256(edges.read_bytes()).hexdigest(),
        "departments": hashlib.sha256(departments.read_bytes()).hexdigest(),
    }

    target = materialize_email_from_files(
        edges_path=edges,
        departments_path=departments,
        runtime_root=tmp_path / "runtime",
        seed=13,
    )

    bundle = load_core_graph_bundle_json((target / "bundle.json").read_bytes())
    manifest = json.loads((target / "materialization-manifest.json").read_text(encoding="utf-8"))
    offline = json.loads((target / "offline-community-labels.json").read_text(encoding="utf-8"))
    assert target.name == "1.0.0"
    assert bundle.source.source_sha256 == manifest["combinedSourceSha256"]
    assert manifest["observedRawSha256"] == source_hashes
    assert manifest["expectedRawSha256"] == {"departments": None, "edges": None}
    assert manifest["outputSemantics"] == "static relation completion"
    assert bundle.node_features == ()
    assert offline["schemaVersion"] == "socialgraph-fm.core-offline-community-labels/1.0"
    assert offline["labels"] == {"department": {"0": "7", "1": "7", "2": "9", "3": "9"}}
    assert len(offline["labelsSha256"]) == 64
    assert len(bundle.split_manifest.assignments) == len(bundle.edges)
    assert len(bundle.edges) == 5
    assert not list((tmp_path / "runtime").rglob("*.staging"))
    assert validate_email_materialization(target) == target


def test_email_reload_rejects_tampered_offline_labels_and_recipe_identity(tmp_path) -> None:
    edges = tmp_path / "edges.txt"
    edges.write_text("0 1\n1 2\n2 0\n", encoding="utf-8")
    departments = tmp_path / "departments.txt"
    departments.write_text("0 7\n1 7\n2 9\n", encoding="utf-8")
    target = materialize_email_from_files(
        edges_path=edges,
        departments_path=departments,
        runtime_root=tmp_path / "runtime",
        seed=13,
    )

    offline_path = target / "offline-community-labels.json"
    offline = json.loads(offline_path.read_text(encoding="utf-8"))
    offline["labels"]["department"]["0"] = "999"
    offline_path.write_text(canonical_json(offline) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="offline label"):
        validate_email_materialization(target)

    offline["labelsSha256"] = canonical_sha256(offline["labels"])
    offline_path.write_text(canonical_json(offline) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bound by materialization"):
        validate_email_materialization(target)

    manifest_path = target / "materialization-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["offlineLabelsSha256"] = offline["labelsSha256"]
    manifest["recipeId"] = "attacker-recipe"
    manifest["manifestSha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifestSha256"}
    )
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe identity"):
        validate_email_materialization(target)


def test_email_reload_rehashes_catalog_raw_files_against_self_consistent_rewrite(
    tmp_path,
) -> None:
    runtime = tmp_path / "runtime"
    edges = tmp_path / "edges.txt"
    edges.write_text("0 1\n1 2\n2 0\n", encoding="utf-8")
    departments = tmp_path / "departments.txt"
    departments.write_text("0 7\n1 7\n2 9\n", encoding="utf-8")
    raw = runtime / "raw" / "email-eu-core" / "1.0.0"
    raw.mkdir(parents=True)
    raw_edges = raw / "email-Eu-core.txt.gz"
    raw_departments = raw / "email-Eu-core-department-labels.txt.gz"
    raw_edges.write_bytes(b"fixed catalog edge source")
    raw_departments.write_bytes(b"fixed catalog department source")
    target = materialize_email_from_files(
        edges_path=edges,
        departments_path=departments,
        raw_source_paths={"edges": raw_edges, "departments": raw_departments},
        runtime_root=runtime,
        seed=13,
    )

    manifest_path = target / "materialization-manifest.json"
    bundle_path = target / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    forged_observed = {
        "departments": hashlib.sha256(b"forged departments").hexdigest(),
        "edges": hashlib.sha256(b"forged edges").hexdigest(),
    }
    forged_combined = canonical_sha256(forged_observed)
    bundle["source"]["sourceSha256"] = forged_combined
    bundle["graphVersionHash"] = calculate_graph_version_hash(bundle)
    bundle_path.write_text(canonical_json(bundle) + "\n", encoding="utf-8")
    manifest["observedRawSha256"] = forged_observed
    manifest["combinedSourceSha256"] = forged_combined
    manifest["graphVersionHash"] = bundle["graphVersionHash"]
    manifest["manifestSha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifestSha256"}
    )
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    assert validate_email_materialization(target) == target

    with pytest.raises(ValueError, match="raw source hash mismatch"):
        materialize_email_eu_core(runtime_root=runtime, seed=13)
