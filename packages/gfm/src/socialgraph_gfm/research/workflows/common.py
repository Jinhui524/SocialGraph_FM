"""Shared schemas, configuration, path safety, and artifact I/O for SocialGraph-FM Research."""

from __future__ import annotations

import json
import os
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from ..routing import route_contract

CORPUS_SCHEMA = "socialgraph-fm.research-corpus/1.0"
TRAINING_SCHEMA = "socialgraph-fm.research-training/1.0"
EVALUATION_SCHEMA = "socialgraph-fm.research-evaluation/1.0"
EXPORT_SCHEMA = "socialgraph-fm.research-export/1.0"
SMOKE_SCHEMA = "socialgraph-fm.research-smoke/1.0"
REGISTRY_SCHEMA = "socialgraph-fm.research-registry/1.0"
MATERIALIZER_VERSION = "research-materializer/1.1.0"
FEATURE_CONTRACT_SCHEMA = "socialgraph-fm.research-feature-contract/1.2"
FRESH_HTTP_STARTUP_TIMEOUT_SECONDS = 60
FRESH_HTTP_RUN_TIMEOUT_SECONDS = 10 * 60

PARSER_CONTRACTS: dict[str, tuple[str, str]] = {
    "twitch-language": ("musae-twitch", "1.0.0"),
    "tolokers": ("tolokers-npz", "1.0.0"),
    "wiki-rfa": ("wiki-rfa-majority-sign", "1.0.0"),
    "email-eu-core": ("email-eu-core-static", "1.0.0"),
}


def _domain_task_id(domain: str) -> str:
    if domain.startswith("twitch-"):
        return CONTENT_POLICY_TASK
    return {
        "tolokers": ACCOUNT_RISK_TASK,
        "wiki-rfa": SIGNED_RELATION_TASK,
        "email-eu-core": COLLABORATION_TASK,
    }[domain]


def _route_contract_hash() -> str:
    return canonical_sha256(route_contract())

TWITCH_ARCHIVE_MEMBERS = frozenset(
    {
        "twitch/DE/musae_DE.json",
        "twitch/DE/musae_DE_edges.csv",
        "twitch/DE/musae_DE_target.csv",
        "twitch/ENGB/musae_ENGB_edges.csv",
        "twitch/ENGB/musae_ENGB_features.json",
        "twitch/ENGB/musae_ENGB_target.csv",
        "twitch/ES/musae_ES_edges.csv",
        "twitch/ES/musae_ES_features.json",
        "twitch/ES/musae_ES_target.csv",
        "twitch/FR/musae_FR_edges.csv",
        "twitch/FR/musae_FR_features.json",
        "twitch/FR/musae_FR_target.csv",
        "twitch/PTBR/musae_PTBR_edges.csv",
        "twitch/PTBR/musae_PTBR_features.json",
        "twitch/PTBR/musae_PTBR_target.csv",
        "twitch/RU/musae_RU_edges.csv",
        "twitch/RU/musae_RU_features.json",
        "twitch/RU/musae_RU_target.csv",
        "twitch/citing.txt",
        "twitch/README.txt",
    }
)

EXPECTED_SOURCE_HASHES: dict[str, str] = {
    "twitch.zip": "65a6c4c23da23889517734a8c947e522b2f0c7db179559a3904aeb8793d004dc",
    "tolokers.npz": "dacf3ac94cec53d03cd2adb5255c08b33dee1656c33ca8164a464bd9450a1667",
    "wiki-RfA.txt.gz": "88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862",
    "email-Eu-core.txt.gz": "4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f",
    "email-Eu-core-department-labels.txt.gz": (
        "e5abe5b4581a480032a63adcf2576c161785f45692642c6ebb0b1276f0f33669"
    ),
}

RESEARCH_SOURCE_RECIPES: dict[str, tuple[str, str, str]] = {
    "twitch.zip": ("twitch-language", "twitch", "raw/twitch-language/1.0.0/twitch.zip"),
    "tolokers.npz": ("tolokers", "tolokers", "raw/tolokers/1.0.0/tolokers.npz"),
    "wiki-RfA.txt.gz": ("wiki-rfa", "wiki-rfa", "raw/wiki-rfa/1.0.0/wiki-RfA.txt.gz"),
    "email-Eu-core.txt.gz": (
        "email-eu-core",
        "edges",
        "raw/email-eu-core/1.0.0/email-Eu-core.txt.gz",
    ),
    "email-Eu-core-department-labels.txt.gz": (
        "email-eu-core",
        "departments",
        "raw/email-eu-core/1.0.0/email-Eu-core-department-labels.txt.gz",
    ),
}


def _research_config_path() -> Path:
    packaged = resources.files("socialgraph_gfm").joinpath(
        "resources/configs/socialgraph-research.json"
    )
    if packaged.is_file():
        return Path(str(packaged))
    source = Path(__file__).resolve().parents[4] / "configs/socialgraph-research.json"
    if source.is_file():
        return source
    raise FileNotFoundError("the pinned SocialGraph-FM Research configuration is unavailable")


def load_research_config() -> dict[str, Any]:
    """Load and strictly validate the single checked-in SocialGraph-FM Research protocol."""

    path = _research_config_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "socialgraph-fm.research-config/1.0":
        raise ValueError("unsupported SocialGraph-FM Research configuration schema")
    if (
        payload.get("releaseId") != RELEASE_ID
        or payload.get("seed") != RESEARCH_SEED
        or payload.get("preliminary") is not True
        or payload.get("formalReadinessUnaffected") is not True
    ):
        raise ValueError("SocialGraph-FM Research configuration identity is invalid")
    model = payload.get("model")
    expected_model = {
        "hiddenDim": 128,
        "encoderLayers": 3,
        "dropout": 0.2,
        "fieldMaskRate": 0.15,
        "edgeMaskRate": 0.1,
        "learningRate": 0.001,
        "weightDecay": 0.0001,
        "pretrainEpochs": 60,
        "pretrainPatience": 8,
        "headEpochs": 100,
        "headPatience": 10,
        "fullBatchEdgeThreshold": 1_500_000,
        "neighborFallback": {"fanout": [20, 10, 5], "batchSize": 2048},
    }
    if model != expected_model:
        raise ValueError("SocialGraph-FM Research model protocol differs from the approved fixed config")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 4:
        raise ValueError("SocialGraph-FM Research configuration requires four dataset families")
    configured_sources = {
        name: digest
        for dataset in datasets
        if isinstance(dataset, dict)
        for name, digest in dataset.get("sourceFiles", {}).items()
    }
    if configured_sources != EXPECTED_SOURCE_HASHES:
        raise ValueError("SocialGraph-FM Research source hashes differ from the approved inventory")
    result = dict(payload)
    result["configSha256"] = file_sha256(path)
    return result


def research_root_from_home(home: str | Path) -> Path:
    selected = Path(home).expanduser().resolve()
    if selected == Path(selected.anchor):
        raise ValueError("research home must not be a filesystem root")
    return selected if selected.name == RELEASE_ID else selected / RELEASE_ID


def _safe_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("research root must not be a filesystem root")
    return root


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_hashed_document(path: Path, *, schema: str, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != schema:
        raise ValueError(f"unsupported artifact schema at {path}")
    observed = payload.get(hash_field)
    expected = canonical_sha256({key: value for key, value in payload.items() if key != hash_field})
    if observed != expected:
        raise ValueError(f"artifact hash mismatch at {path}")
    return payload

COMPAT_EXPORTS = (
    'CORPUS_SCHEMA',
    'TRAINING_SCHEMA',
    'EVALUATION_SCHEMA',
    'EXPORT_SCHEMA',
    'SMOKE_SCHEMA',
    'REGISTRY_SCHEMA',
    'MATERIALIZER_VERSION',
    'FEATURE_CONTRACT_SCHEMA',
    'FRESH_HTTP_STARTUP_TIMEOUT_SECONDS',
    'FRESH_HTTP_RUN_TIMEOUT_SECONDS',
    'PARSER_CONTRACTS',
    '_domain_task_id',
    '_route_contract_hash',
    'TWITCH_ARCHIVE_MEMBERS',
    'EXPECTED_SOURCE_HASHES',
    'RESEARCH_SOURCE_RECIPES',
    '_research_config_path',
    'load_research_config',
    'research_root_from_home',
    '_safe_root',
    '_atomic_json',
    '_read_hashed_document',
)

__all__ = [
    'load_research_config',
    'research_root_from_home',
]
