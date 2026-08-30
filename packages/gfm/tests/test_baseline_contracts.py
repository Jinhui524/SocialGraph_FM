from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.contracts import (
    BaselineAcceptanceReport,
    BaselineConfig,
    BaselineEvaluationReport,
    CorpusArrayManifest,
    FormalCorpusManifest,
    TemporalLinkProtocolManifest,
)


SHA = "a" * 64


def _corpus_payload(path: str) -> dict:
    array = CorpusArrayManifest(
        name="x",
        sha256=SHA,
        dtype="float32",
        shape=(235868, 128),
        byteCount=235868 * 128 * 4,
    )
    protocol = TemporalLinkProtocolManifest(
        protocolId="ogbl-collab-official-v1",
        evaluator="ogb.linkproppred.Evaluator(ogbl-collab)",
    )
    payload = {
        "schemaVersion": "gfm.formal-corpus/1.0",
        "corpusId": "ogbl-collab",
        "purpose": "formal_benchmark",
        "datasetRole": "benchmark",
        "ogbVersion": "1.3.6",
        "licenseId": "ODC-BY-1.0",
        "licenseSourceUrl": "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab",
        "licenseAccepted": True,
        "attribution": "Open Graph Benchmark: ogbl-collab",
        "sourceFingerprint": SHA,
        "packageSha256": SHA,
        "adapter": "socialgraph-fm-api.convert-ogbl-collab",
        "adapterVersion": "1.0",
        "nodeCount": 235868,
        "featureShape": (235868, 128),
        "messageEdgeCount": 100,
        "splitSizes": {"train": 10, "validation": 2, "test": 3},
        "arrays": (array,),
        "temporalProtocol": protocol,
        "warnings": ("feature time is not verifiable",),
    }
    return {
        **payload,
        "logicalHash": canonical_sha256(payload),
        "createdAt": datetime(2026, 8, 12, tzinfo=UTC),
        "artifactPath": path,
    }


def test_formal_corpus_hash_is_independent_of_absolute_artifact_path():
    first = FormalCorpusManifest.model_validate(_corpus_payload(r"C:\one"))
    second = FormalCorpusManifest.model_validate(_corpus_payload(r"E:\two"))
    assert first.logical_hash == second.logical_hash
    assert first.logical_payload() == second.logical_payload()

    tampered = _corpus_payload(r"E:\two")
    tampered["messageEdgeCount"] = 101
    with pytest.raises(ValidationError, match="logicalHash"):
        FormalCorpusManifest.model_validate(tampered)


def test_baseline_v1_config_is_decision_complete_and_immutable():
    config = BaselineConfig(
        configId="ogbl-collab-baseline",
        tracks=("ogb_official", "strict_edge_time"),
        models=("cn", "aa", "ra", "mlp", "graphsage"),
        formalSeeds=(20260812, 20260813, 20260814),
        neighborFanout=(15, 10),
        candidateBatchSizes=(4096, 2048, 1024),
    )
    assert config.formal_max_epochs == 50
    with pytest.raises(ValidationError, match="fixed model suite"):
        BaselineConfig.model_validate(
            {**config.model_dump(by_alias=True), "models": ["mlp", "graphsage"]}
        )


def test_evaluation_contract_forbids_test_access_during_dev():
    payload = {
        "schemaVersion": "gfm.baseline-evaluation/1.0",
        "experimentId": "exp",
        "runId": "run",
        "phase": "dev",
        "track": "ogb_official",
        "model": "mlp",
        "seed": 20260811,
        "validationMetrics": {"Hits@50": 0.1},
        "testMetrics": {"Hits@50": 0.1},
        "strata": {},
        "scoreCounts": {"validationPositive": 1},
        "testReadAfterSelection": True,
    }
    payload["reportHash"] = canonical_sha256(payload)
    with pytest.raises(ValidationError, match="must not read test"):
        BaselineEvaluationReport.model_validate(payload)


def test_acceptance_cannot_be_true_with_an_unmet_gate():
    payload = {
        "schemaVersion": "gfm.baseline-acceptance/1.0",
        "experimentId": "exp",
        "accepted": True,
        "corpusHash": SHA,
        "configHash": SHA,
        "requiredLearningRuns": 12,
        "completedLearningRuns": 12,
        "completedHeuristicRuns": 6,
        "peakCudaMemoryMiB": 1024.0,
        "metricSummary": {"graphsage": {"testMean": 0.4}},
        "gates": {"quality": False},
        "warnings": (),
    }
    payload["reportHash"] = canonical_sha256(payload)
    payload["createdAt"] = datetime(2026, 8, 12, tzinfo=UTC)
    with pytest.raises(ValidationError, match="all fixed gates"):
        BaselineAcceptanceReport.model_validate(payload)

