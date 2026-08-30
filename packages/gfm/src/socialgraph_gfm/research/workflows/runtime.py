"""Checkpoint reconstruction and trained-artifact validation shared by workflow stages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    SIGNED_RELATION_TASK,
)
from ..routing import route_contract
from .common import (
    TRAINING_SCHEMA,
    _read_hashed_document,
    _route_contract_hash,
    load_research_config,
)
from .materialize import load_corpus_manifest


def _checkpoint_runtime(root: Path, checkpoint: Mapping[str, Any], *, device: str):
    from ...core.adapters import AdapterSchema, BundleInputAdapter
    from ...core.model import ResearchCoreGFM
    from .train import _load_graph_documents, _tensor_state_hash

    config_sha = load_research_config()["configSha256"]
    if checkpoint.get("researchConfigSha256") != config_sha:
        raise ValueError("research checkpoint configuration identity mismatch")
    if (
        checkpoint.get("routeContract") != route_contract()
        or checkpoint.get("routeContractHash") != _route_contract_hash()
    ):
        raise ValueError("research checkpoint route contract mismatch")
    corpus = load_corpus_manifest(root)
    if checkpoint.get("corpusHash") != corpus["corpusHash"]:
        raise ValueError("research checkpoint corpus identity mismatch")
    if checkpoint.get("schemaVersion") == "socialgraph-fm.research-serving-checkpoint/1.0" and (
        checkpoint.get("corpusKind") != corpus["corpusKind"]
        or checkpoint.get("testOnly") is not corpus["testOnly"]
    ):
        raise ValueError("research serving checkpoint corpus kind mismatch")
    calibrators = checkpoint.get("calibrators")
    if not isinstance(calibrators, Mapping) or set(calibrators) != {
        CONTENT_POLICY_TASK,
        ACCOUNT_RISK_TASK,
        SIGNED_RELATION_TASK,
        COLLABORATION_TASK,
    }:
        raise ValueError("research checkpoint calibrator inventory mismatch")
    for task_id, calibrator in calibrators.items():
        if calibrator.get("taskId") != task_id or calibrator.get(
            "artifactHash"
        ) != canonical_sha256(
            {key: value for key, value in calibrator.items() if key != "artifactHash"}
        ):
            raise ValueError("research checkpoint calibrator identity mismatch")
    fold_calibrators = checkpoint.get("tolokersFoldCalibrators")
    if not isinstance(fold_calibrators, (list, tuple)) or len(fold_calibrators) != 10:
        raise ValueError("research checkpoint lacks ten Tolokers split calibrators")
    if tuple(item.get("fold") for item in fold_calibrators) != tuple(range(10)):
        raise ValueError("Tolokers split calibrator order mismatch")
    for item in fold_calibrators:
        if item.get("wrapperHash") != canonical_sha256(
            {key: value for key, value in item.items() if key != "wrapperHash"}
        ):
            raise ValueError("Tolokers split calibrator wrapper hash mismatch")
    documents = _load_graph_documents(root, corpus)
    domains = tuple(checkpoint["domains"])
    if domains != tuple(sorted(documents)):
        raise ValueError("research checkpoint domain inventory mismatch")
    model = ResearchCoreGFM(domains=domains).to(device)
    model.load_state_dict(checkpoint["modelState"], strict=True)
    if checkpoint.get("modelStateHash") not in {
        None,
        _tensor_state_hash(model.state_dict()),
    }:
        raise ValueError("research checkpoint model state identity mismatch")
    model.eval()
    adapters = {}
    for domain in domains:
        schema = AdapterSchema.model_validate_json(
            canonical_json(checkpoint["adapterSchemas"][domain])
        )
        adapter = BundleInputAdapter(documents[domain][0], schema=schema, mode="training").to(
            device
        )
        adapter.load_state_dict(checkpoint["adapterStates"][domain], strict=True)
        expected_adapter_hashes = checkpoint.get("adapterStateHashes") or {}
        if expected_adapter_hashes.get(domain) not in {
            None,
            _tensor_state_hash(adapter.state_dict()),
        }:
            raise ValueError("research checkpoint adapter state identity mismatch")
        adapter.eval()
        adapters[domain] = adapter
    return corpus, documents, model, adapters

def _load_trained_runtime(root: Path, *, device: str):
    import torch

    from .train import _tensor_state_hash

    training = _read_hashed_document(
        root / "runs/shared/training-manifest.json",
        schema=TRAINING_SCHEMA,
        hash_field="trainingHash",
    )
    config_sha = load_research_config()["configSha256"]
    if training.get("researchConfigSha256") != config_sha:
        raise ValueError("research training configuration identity mismatch")
    if (
        training.get("routeContract") != route_contract()
        or training.get("routeContractHash") != _route_contract_hash()
    ):
        raise ValueError("research training route contract mismatch")
    checkpoint_path = root / "runs/shared" / training["checkpointPath"]
    if file_sha256(checkpoint_path) != training["checkpointSha256"]:
        raise ValueError("research training checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schemaVersion") != "socialgraph-fm.research-checkpoint/1.0":
        raise ValueError("unsupported research checkpoint schema")
    corpus, documents, model, adapters = _checkpoint_runtime(root, checkpoint, device=device)
    if training.get("corpusHash") != corpus["corpusHash"]:
        raise ValueError("research training corpus identity mismatch")
    if training.get("modelStateHash") != _tensor_state_hash(model.state_dict()):
        raise ValueError("research training model state identity mismatch")
    if training.get("adapterSchemaHashes") != {
        domain: adapters[domain].schema.adapter_schema_hash for domain in sorted(adapters)
    }:
        raise ValueError("research training adapter schema identity mismatch")
    return training, checkpoint, corpus, documents, model, adapters

COMPAT_EXPORTS = ("_checkpoint_runtime", "_load_trained_runtime")

__all__: list[str] = []
