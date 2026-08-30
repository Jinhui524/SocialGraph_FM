"""Pinned SocialGraph-FM Global task, protocol and runtime configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.identity import code_identity_hash
from socialgraph_gfm.locks import verify_lock_manifest

RELEASE_ID = "socialgraph-fm"
RUNTIME_DIRECTORY = "socialgraph-global"
TASK_ID = "coordination_risk"
SEED = 12121995
SPLIT_INDEX = 0
COUNTRIES = ("china", "cuba", "iran", "russia", "UAE", "venezuela")
SOURCE_COUNTRIES = ("china", "cuba", "iran", "UAE", "venezuela")
ProtocolId = Literal["in_domain", "low_label", "cross_domain", "global"]
SplitName = Literal["train", "validation", "test"]
PINNED_MODEL = {
    "textDim": 768,
    "structuralDim": 128,
    "branchDim": 128,
    "hiddenDim": 256,
    "gnnLayers": 2,
    "dropout": 0.2,
    "routerEnabled": True,
    "routerBottleneckDim": 64,
    "routerExperts": 8,
    "routerTopK": 2,
}
PINNED_TRAINING = {
    "maxSteps": 1000,
    "minSteps": 100,
    "evalEverySteps": 25,
    "patienceEvals": 8,
    "learningRate": 0.001,
    "weightDecay": 0.0,
    "seedBatchSize": 128,
    "numNeighbors": [20, 10],
    "memorySmokeBatchSizes": [256, 128, 64],
    "maxPeakMiB": 5632,
    "gradientClipNorm": 1.0,
    "routerBalanceWeight": 0.01,
    "amp": True,
    "checkpointEverySteps": 25,
    "numWorkers": 0,
}
PINNED_SELECTION = {
    "primaryMetric": "country-balanced-macro-f1",
    "thresholdMetric": "country-balanced-macro-f1",
    "calibration": "binary-logit",
}


@dataclass(frozen=True)
class DatasetRef:
    country: str
    variant: str
    split: SplitName

    @classmethod
    def from_dict(cls, value: Any) -> DatasetRef:
        if not isinstance(value, dict) or set(value) != {"country", "variant", "split"}:
            raise ValueError("Global dataset reference must contain country, variant and split")
        country = value["country"]
        variant = value["variant"]
        split = value["split"]
        if country not in COUNTRIES:
            raise ValueError(f"unsupported Global country: {country}")
        if variant not in {"base", "0.95U"}:
            raise ValueError(f"unsupported Global split variant: {variant}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported Global split: {split}")
        return cls(country=country, variant=variant, split=split)

    def to_dict(self) -> dict[str, str]:
        return {"country": self.country, "variant": self.variant, "split": self.split}


@dataclass(frozen=True)
class ProtocolPlan:
    protocol: ProtocolId
    train: tuple[DatasetRef, ...]
    select: tuple[DatasetRef, ...]
    calibrate: tuple[DatasetRef, ...]
    evaluate: tuple[DatasetRef, ...]
    target_policy: str | None = None

    @property
    def train_domains(self) -> tuple[str, ...]:
        return tuple(item.country for item in self.train)

    @property
    def selection_domains(self) -> tuple[str, ...]:
        return tuple(item.country for item in self.select)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "protocol": self.protocol,
            "train": [item.to_dict() for item in self.train],
            "select": [item.to_dict() for item in self.select],
            "calibrate": [item.to_dict() for item in self.calibrate],
            "evaluate": [item.to_dict() for item in self.evaluate],
        }
        if self.target_policy is not None:
            result["targetPolicy"] = self.target_policy
        return result


def _config_path() -> Path:
    packaged = resources.files("socialgraph_gfm").joinpath(
        "resources/configs/socialgraph-global.json"
    )
    if packaged.is_file():
        return Path(str(packaged))
    source = Path(__file__).resolve().parents[3] / "configs/socialgraph-global.json"
    if source.is_file():
        return source
    raise FileNotFoundError("the pinned SocialGraph-FM Global configuration is unavailable")


def _refs(values: Any) -> tuple[DatasetRef, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("Global protocol stages require a nonempty dataset inventory")
    return tuple(DatasetRef.from_dict(value) for value in values)


def _protocol(protocol: ProtocolId, value: Any) -> ProtocolPlan:
    if not isinstance(value, dict):
        raise TypeError(f"Global protocol {protocol} must be an object")
    expected_keys = {"train", "select", "calibrate", "evaluate"}
    extra_keys = set(value) - expected_keys - {"targetPolicy"}
    if extra_keys or not expected_keys.issubset(value):
        raise ValueError(f"Global protocol {protocol} has an invalid stage inventory")
    result = ProtocolPlan(
        protocol=protocol,
        train=_refs(value["train"]),
        select=_refs(value["select"]),
        calibrate=_refs(value["calibrate"]),
        evaluate=_refs(value["evaluate"]),
        target_policy=value.get("targetPolicy"),
    )
    _validate_protocol(result)
    return result


def _validate_protocol(plan: ProtocolPlan) -> None:
    for item in plan.train:
        if item.split != "train":
            raise ValueError(f"{plan.protocol} training may consume train masks only")
    for item in (*plan.select, *plan.calibrate):
        if item.split != "validation" or item.variant != "base":
            raise ValueError(f"{plan.protocol} selection/calibration must use base validation")
    for item in plan.evaluate:
        if item.split != "test" or item.variant != "base":
            raise ValueError(f"{plan.protocol} evaluation must use base test")
    if plan.protocol == "in_domain":
        expected = ("russia",)
        if (
            plan.train_domains != expected
            or plan.selection_domains != expected
            or tuple(item.country for item in plan.evaluate) != expected
            or plan.train[0].variant != "base"
        ):
            raise ValueError("In-domain must train, select and evaluate on the base Russia split")
    elif plan.protocol == "low_label":
        if (
            plan.train_domains != ("russia",)
            or plan.train[0].variant != "0.95U"
            or plan.selection_domains != ("russia",)
            or tuple(item.country for item in plan.evaluate) != ("russia",)
        ):
            raise ValueError("Low-label must train on Russia 0.95U and use base validation/test")
    elif plan.protocol == "cross_domain":
        if (
            plan.train_domains != SOURCE_COUNTRIES
            or plan.selection_domains != SOURCE_COUNTRIES
            or tuple(item.country for item in plan.calibrate) != SOURCE_COUNTRIES
            or tuple(item.country for item in plan.evaluate) != ("russia",)
            or plan.target_policy != "source-only-selection-then-single-frozen-target-test"
        ):
            raise ValueError("Cross-domain must be source-only until one frozen Russia test evaluation")
        pre_freeze = (*plan.train, *plan.select, *plan.calibrate)
        if any(item.country == "russia" for item in pre_freeze):
            raise ValueError("Cross-domain Russia labels or masks are forbidden before model freeze")
    elif any(
        tuple(item.country for item in stage) != COUNTRIES
        for stage in (plan.train, plan.select, plan.calibrate, plan.evaluate)
    ):
        raise ValueError("global protocol must cover all six countries in every stage")


def load_global_model_config() -> dict[str, Any]:
    """Load and validate the immutable SocialGraph-FM Global release configuration."""

    path = _config_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schemaVersion",
        "releaseId",
        "taskId",
        "seed",
        "splitIndex",
        "model",
        "training",
        "selection",
        "protocols",
    }:
        raise ValueError("SocialGraph-FM Global configuration has an unexpected top-level shape")
    if (
        payload["schemaVersion"] != "socialgraph-fm.global-model-config/1.0"
        or payload["releaseId"] != RELEASE_ID
        or payload["taskId"] != TASK_ID
        or payload["seed"] != SEED
        or payload["splitIndex"] != SPLIT_INDEX
    ):
        raise ValueError("SocialGraph-FM Global configuration identity is invalid")
    if (
        payload["model"] != PINNED_MODEL
        or payload["training"] != PINNED_TRAINING
        or payload["selection"] != PINNED_SELECTION
    ):
        raise ValueError("SocialGraph-FM Global model, training or selection protocol is not pinned")
    raw_protocols = payload["protocols"]
    if not isinstance(raw_protocols, dict) or set(raw_protocols) != {
        "in_domain",
        "low_label",
        "cross_domain",
        "global",
    }:
        raise ValueError("SocialGraph-FM Global requires exactly in_domain, low_label, cross_domain and global protocols")
    protocol_ids: tuple[ProtocolId, ...] = ("in_domain", "low_label", "cross_domain", "global")
    plans = {
        protocol: _protocol(protocol, raw_protocols[protocol])
        for protocol in protocol_ids
    }
    result = dict(payload)
    result["configSha256"] = file_sha256(path)
    result["protocolPlans"] = plans
    return result


def protocol_plan(protocol: ProtocolId) -> ProtocolPlan:
    return load_global_model_config()["protocolPlans"][protocol]


def global_model_root_from_home(home: str | Path) -> Path:
    selected = Path(home).expanduser().resolve()
    if selected == Path(selected.anchor):
        raise ValueError("Global home must not be a filesystem root")
    return selected if selected.name == RUNTIME_DIRECTORY else selected / RUNTIME_DIRECTORY


def release_identity() -> dict[str, str]:
    """Bind runs to checked-in code, configuration and the runtime lock manifest."""

    config = load_global_model_config()
    locks = verify_lock_manifest()
    manifest = Path(locks["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError("the Global runtime lock manifest is unavailable")
    identity = {
        "codeHash": code_identity_hash(),
        "configHash": config["configSha256"],
        "runtimeLockHash": file_sha256(manifest),
    }
    identity["releaseIdentityHash"] = canonical_sha256(identity)
    return identity
