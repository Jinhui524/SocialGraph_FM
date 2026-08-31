from pathlib import Path

from socialgraph_gfm.core_contracts import CoreReadiness

REPOSITORY = Path(__file__).resolve().parents[3]


def test_package_readiness_is_the_current_core_machine_truth() -> None:
    package_record = REPOSITORY / "packages" / "gfm" / "contracts" / "core-readiness.json"
    readiness = CoreReadiness.model_validate_json(package_record.read_bytes())

    assert readiness.milestone == "SocialGraph-FM Core"
    assert readiness.identity.control_generation == 0
    assert readiness.identity.registry_generation == 0
    assert readiness.evidence.preflight_evidence_hash is None
    assert readiness.evidence.acceptance_hash is None
    assert readiness.evidence.accepted_candidate_hash is None
    assert readiness.evidence.serving_smoke_hash is None
    assert readiness.evidence.serving_model_hash is None
    assert readiness.gates.corpus_ready.ready is False
    assert readiness.gates.corpus_ready.reason_code == "FORMAL_PREFLIGHT_MISSING"
    assert readiness.gates.model_validated.ready is False
    assert readiness.gates.accepted.ready is False
    assert readiness.gates.core_serving_ready.ready is False


def test_public_documentation_has_one_chinese_readme_and_reference() -> None:
    docs_root = REPOSITORY / "docs"
    expected_files = {"REFERENCE.md", "status/readiness.json"}
    actual_files = {
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files

    reference = (docs_root / "REFERENCE.md").read_text(encoding="utf-8")
    assert "SocialGraph-FM" in reference
    assert "一个受管 Python" in reference
    assert (docs_root / "status" / "readiness.json").read_bytes() == (
        REPOSITORY / "packages" / "gfm" / "contracts" / "core-readiness.json"
    ).read_bytes()
    readme = REPOSITORY / "README.md"
    assert readme.is_file()
    assert not (REPOSITORY / "README.zh-CN.md").exists()
    assert "三步启动" in readme.read_text(encoding="utf-8")


def test_removed_knowledge_source_handoff_and_history_documents_stay_absent() -> None:
    removed_paths = (
        REPOSITORY / "docs" / "knowledge-sources.json",
        REPOSITORY / "docs" / "archive" / "history.md",
        REPOSITORY
        / "docs"
        / "model-and-data"
        / "core"
        / "EXPERIMENT_RUNBOOK.md",
        REPOSITORY / "docs" / "model-and-data" / "core" / "MODEL_CARD.md",
        REPOSITORY / "docs" / "model-and-data" / "core" / "DATASET_CARDS.md",
    )
    assert all(not path.exists() for path in removed_paths)
