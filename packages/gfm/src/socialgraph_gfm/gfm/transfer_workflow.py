"""Formal transfer isolation and architecture-promotion decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard

import numpy as np
from torch import Tensor

from ..canonical import canonical_sha256
from .model import SocialGraphFMCore


DOMAIN_FAMILY_BY_ID = {
    "openalex-graph-ai": "academic-collaboration",
    "thgl-software-2.0.0": "software-activity",
    "wikimedia-talk-article-2011-2015": "online-community",
}
FORMAL_DOMAIN_IDS = frozenset(DOMAIN_FAMILY_BY_ID)
FORMAL_SEEDS = (20260821, 20260822, 20260823)
_SHA256_HEX = frozenset("0123456789abcdef")


def _is_lower_sha256(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256_HEX)


def _checked_hashes(values: Sequence[str], *, name: str, minimum: int = 1) -> tuple[str, ...]:
    checked = tuple(values)
    if len(checked) < minimum or len(set(checked)) != len(checked):
        raise ValueError(f"{name} must contain at least {minimum} unique hashes")
    if any(not _is_lower_sha256(value) for value in checked):
        raise ValueError(f"{name} must contain lowercase SHA-256 values")
    return checked


@dataclass(frozen=True)
class LodoIsolationAudit:
    held_out_family: str
    source_domain_ids: tuple[str, ...]
    target_domain_ids: tuple[str, ...]
    source_corpus_hashes: tuple[str, ...]
    target_corpus_hashes: tuple[str, ...]
    verified_corpus_hashes: tuple[str, ...]
    adapter_statistic_hashes: tuple[str, ...]
    excluded_academic_sibling_ids: tuple[str, ...]
    academic_sibling_access_count: int
    academic_sibling_exclusion_evidence_hash: str
    target_adapter_initialized_after_pretraining: bool

    def validate(self) -> None:
        if self.held_out_family not in set(DOMAIN_FAMILY_BY_ID.values()):
            raise ValueError("LODO held-out family is unknown")
        if (
            len(self.source_domain_ids) != 2
            or len(set(self.source_domain_ids)) != 2
            or len(self.target_domain_ids) != 1
        ):
            raise ValueError("LODO requires exactly two source domains and one target domain")
        source_ids = set(self.source_domain_ids)
        target_ids = set(self.target_domain_ids)
        if source_ids.intersection(target_ids):
            raise ValueError("LODO target domain appeared among source domains")
        source_families = {DOMAIN_FAMILY_BY_ID.get(value) for value in source_ids}
        target_families = {DOMAIN_FAMILY_BY_ID.get(value) for value in target_ids}
        if None in source_families or None in target_families:
            raise ValueError("LODO domain ID has no registered family")
        expected_sources = {
            domain_id
            for domain_id, family in DOMAIN_FAMILY_BY_ID.items()
            if family != self.held_out_family
        }
        expected_targets = {
            domain_id
            for domain_id, family in DOMAIN_FAMILY_BY_ID.items()
            if family == self.held_out_family
        }
        if (
            source_ids != expected_sources
            or target_ids != expected_targets
            or source_families
            != set(DOMAIN_FAMILY_BY_ID.values()).difference({self.held_out_family})
            or target_families != {self.held_out_family}
        ):
            raise ValueError("LODO domain-family isolation failed")
        source_hashes = _checked_hashes(
            self.source_corpus_hashes,
            name="source_corpus_hashes",
            minimum=len(self.source_domain_ids),
        )
        target_hashes = _checked_hashes(
            self.target_corpus_hashes,
            name="target_corpus_hashes",
            minimum=len(self.target_domain_ids),
        )
        verified_hashes = _checked_hashes(
            self.verified_corpus_hashes,
            name="verified_corpus_hashes",
            minimum=len(source_hashes) + len(target_hashes),
        )
        if set(source_hashes).intersection(target_hashes):
            raise ValueError("LODO target corpus hash appeared in pretraining")
        if set(verified_hashes) != set(source_hashes).union(target_hashes):
            raise ValueError("LODO corpus hashes are not exactly the verified corpus inventory")
        _checked_hashes(
            self.adapter_statistic_hashes,
            name="adapter_statistic_hashes",
        )
        siblings = tuple(self.excluded_academic_sibling_ids)
        if (
            not siblings
            or len(set(siblings)) != len(siblings)
            or any(not value or value in FORMAL_DOMAIN_IDS for value in siblings)
        ):
            raise ValueError("LODO requires explicit unique academic sibling exclusions")
        if (
            isinstance(self.academic_sibling_access_count, bool)
            or not isinstance(self.academic_sibling_access_count, int)
            or self.academic_sibling_access_count != 0
        ):
            raise ValueError("LODO academic sibling access count must be exactly zero")
        if not _is_lower_sha256(self.academic_sibling_exclusion_evidence_hash):
            raise ValueError("LODO academic sibling exclusion evidence must be a SHA-256")
        if self.academic_sibling_exclusion_evidence_hash in set(
            source_hashes + target_hashes + self.adapter_statistic_hashes
        ):
            raise ValueError("LODO academic sibling evidence must have an independent identity")
        if self.target_adapter_initialized_after_pretraining is not True:
            raise ValueError("LODO target adapter was initialized before isolation")


def assert_lodo_isolation(audit: LodoIsolationAudit) -> str:
    audit.validate()
    return canonical_sha256(
        {
            "schemaVersion": "gfm.lodo-isolation-audit/1.0",
            "heldOutFamily": audit.held_out_family,
            "sourceDomainIds": sorted(audit.source_domain_ids),
            "targetDomainIds": sorted(audit.target_domain_ids),
            "sourceCorpusHashes": sorted(audit.source_corpus_hashes),
            "targetCorpusHashes": sorted(audit.target_corpus_hashes),
            "verifiedCorpusHashes": sorted(audit.verified_corpus_hashes),
            "adapterStatisticHashes": sorted(audit.adapter_statistic_hashes),
            "excludedAcademicSiblingIds": sorted(audit.excluded_academic_sibling_ids),
            "academicSiblingAccessCount": audit.academic_sibling_access_count,
            "academicSiblingExclusionEvidenceHash": (
                audit.academic_sibling_exclusion_evidence_hash
            ),
            "targetAdapterInitializedAfterPretraining": (
                audit.target_adapter_initialized_after_pretraining
            ),
        }
    )


def few_shot_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    fraction: float,
    seed: int,
    split_roles: Sequence[str] | np.ndarray,
    event_times: Sequence[int | float] | np.ndarray,
    cutoff_time: int | float,
    sample_corpus_hashes: Sequence[str] | np.ndarray,
    expected_corpus_hash: str,
) -> np.ndarray:
    """Select deterministic train-only few-shot rows bound to one corpus and cutoff."""

    if fraction not in (0.01, 0.05, 0.1):
        raise ValueError("formal few-shot fraction must be 1%, 5% or 10%")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) <= 2**32 - 1
    ):
        raise ValueError("formal few-shot seed must be an unsigned 32-bit integer")
    value = np.asarray(labels)
    roles = np.asarray(split_roles)
    times = np.asarray(event_times)
    corpus_hashes = np.asarray(sample_corpus_hashes)
    if (
        value.ndim != 1
        or roles.ndim != 1
        or times.ndim != 1
        or corpus_hashes.ndim != 1
        or not value.size
        or not (value.shape == roles.shape == times.shape == corpus_hashes.shape)
    ):
        raise ValueError("few-shot labels and provenance arrays must be aligned vectors")
    if value.dtype.hasobject or not (
        np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError("few-shot labels must be a nonempty numeric vector")
    if times.dtype.hasobject or not np.issubdtype(times.dtype, np.number):
        raise ValueError("few-shot event times must be numeric")
    if (
        isinstance(cutoff_time, bool)
        or not isinstance(cutoff_time, (int, float, np.integer, np.floating))
        or not np.isfinite(cutoff_time)
    ):
        raise ValueError("few-shot cutoff must be a finite numeric value")
    if not _is_lower_sha256(expected_corpus_hash):
        raise ValueError("few-shot expected corpus identity must be a lowercase SHA-256")
    raw_roles = tuple(roles.tolist())
    if any(
        not isinstance(role, str) or role not in {"train", "validation", "test", "shadow"}
        for role in raw_roles
    ):
        raise ValueError("few-shot split roles contain an unknown or non-string value")
    raw_hashes = tuple(corpus_hashes.tolist())
    if any(not isinstance(value_hash, str) for value_hash in raw_hashes) or any(
        value_hash != expected_corpus_hash for value_hash in raw_hashes
    ):
        raise ValueError("few-shot samples are not bound to the expected corpus")
    if not bool(np.isfinite(times).all()):
        raise ValueError("few-shot event times must be finite")
    train_mask = roles == "train"
    if not bool(train_mask.any()):
        raise ValueError("few-shot selection requires a nonempty train split")
    if bool(np.any(times[train_mask] > cutoff_time)):
        raise ValueError("few-shot train split contains an event after the cutoff")
    train_values = value[train_mask]
    if not bool(np.isfinite(train_values).all()) or not bool(
        np.equal(train_values, np.floor(train_values)).all()
    ):
        raise ValueError("few-shot train labels must be finite discrete values")
    train_indices = np.flatnonzero(train_mask)
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for label in np.unique(train_values):
        candidates = train_indices[train_values == label]
        count = max(1, int(np.floor(candidates.size * float(fraction))))
        if count >= candidates.size:
            chosen = candidates.copy()
        else:
            chosen = np.sort(rng.choice(candidates, size=count, replace=False))
        selected.append(chosen)
    result = np.sort(np.concatenate(selected)).astype(np.int64, copy=False)
    if result.size != np.unique(result).size:
        raise RuntimeError("few-shot selection produced duplicate indices")
    return result


@dataclass(frozen=True)
class VariantSelection:
    selected: Literal["core-base", "core-moe"]
    mean_relative_gain: float
    maximum_domain_regression: float
    moe_promoted: bool
    reason: str


def select_core_variant(
    *,
    base_by_domain: Mapping[str, float],
    moe_by_domain: Mapping[str, float],
    minimum_mean_relative_gain: float = 0.02,
    maximum_domain_relative_regression: float = 0.01,
) -> VariantSelection:
    """Promote MoE only when both fixed gain and no-harm rules pass."""

    if (
        set(base_by_domain) != FORMAL_DOMAIN_IDS
        or set(moe_by_domain) != FORMAL_DOMAIN_IDS
        or minimum_mean_relative_gain != 0.02
        or maximum_domain_relative_regression != 0.01
    ):
        raise ValueError("Core variant selection requires the three fixed domain IDs")
    gains: list[float] = []
    regressions: list[float] = []
    for domain in sorted(base_by_domain):
        base, moe = float(base_by_domain[domain]), float(moe_by_domain[domain])
        if (
            not np.isfinite(base)
            or not np.isfinite(moe)
            or not 0.0 <= base <= 1.0
            or not 0.0 <= moe <= 1.0
        ):
            raise ValueError("variant metrics must be finite values in [0, 1]")
        gain = (moe - base) / max(abs(base), 1e-12)
        gains.append(gain)
        regressions.append(max(0.0, -gain))
    mean_gain = float(np.mean(gains))
    max_regression = float(max(regressions))
    promoted = mean_gain >= 0.02 and max_regression <= 0.01
    return VariantSelection(
        selected="core-moe" if promoted else "core-base",
        mean_relative_gain=mean_gain,
        maximum_domain_regression=max_regression,
        moe_promoted=promoted,
        reason=(
            "MoE met the fixed +2% mean and <=1% per-domain regression gates"
            if promoted
            else "MoE did not meet both fixed promotion gates; core-base remains formal"
        ),
    )


def select_formal_checkpoints(
    records: Sequence[Mapping[str, object]],
    *,
    selected_variant: Literal["core-base", "core-moe"],
    expected_config_hash: str,
    expected_code_hash: str,
    expected_environment_hash: str,
    expected_corpus_hashes: Sequence[str],
    seeds: Sequence[int] = FORMAL_SEEDS,
) -> tuple[str, ...]:
    """Require one fully provenance-bound formal best checkpoint per frozen seed."""

    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in seeds):
        raise ValueError("Core formal seeds must be integers")
    expected = tuple(int(value) for value in seeds)
    if expected != FORMAL_SEEDS:
        raise ValueError("Core formal checkpoint selection uses the three frozen seeds")
    if selected_variant not in ("core-base", "core-moe"):
        raise ValueError("unknown formal architecture variant")
    for name, hash_value in (
        ("expected_config_hash", expected_config_hash),
        ("expected_code_hash", expected_code_hash),
        ("expected_environment_hash", expected_environment_hash),
    ):
        if not _is_lower_sha256(hash_value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    expected_corpora = _checked_hashes(
        expected_corpus_hashes,
        name="expected_corpus_hashes",
        minimum=3,
    )
    if len(records) != len(expected):
        raise ValueError("formal checkpoint coverage is incomplete or contains extra records")
    result: list[str] = []
    digests: list[str] = []
    for seed in expected:
        matches = [
            record
            for record in records
            if record.get("variant") == selected_variant
            and isinstance(record.get("seed"), int)
            and record["seed"] == seed
        ]
        if len(matches) != 1:
            raise ValueError("formal checkpoint coverage is incomplete or ambiguous")
        record = matches[0]
        checkpoint = record.get("checkpointId")
        digest = record.get("freshProcessDigest")
        if (
            not isinstance(checkpoint, str)
            or not checkpoint
            or "-best-" not in checkpoint
            or not _is_lower_sha256(digest)
            or record.get("freshProcessVerified") is not True
            or record.get("checkpointRole") != "best"
            or record.get("phase") != "formal"
            or record.get("configHash") != expected_config_hash
            or record.get("codeHash") != expected_code_hash
            or record.get("environmentHash") != expected_environment_hash
        ):
            raise ValueError("formal checkpoint lacks verified best-state provenance")
        raw_corpora = record.get("corpusHashes")
        if not isinstance(raw_corpora, (list, tuple)):
            raise ValueError("formal checkpoint lacks corpus provenance")
        record_corpora = _checked_hashes(
            raw_corpora,
            name="formal checkpoint corpusHashes",
            minimum=3,
        )
        if set(record_corpora) != set(expected_corpora):
            raise ValueError("formal checkpoint corpus provenance differs")
        result.append(checkpoint)
        digests.append(digest)
    if len(set(result)) != len(result):
        raise ValueError("formal checkpoint IDs must be unique across seeds")
    if len(set(digests)) != len(digests):
        raise ValueError("fresh-process digests must be unique across formal checkpoints")
    return tuple(result)


def load_lodo_shared_backbone(
    target: SocialGraphFMCore,
    source_state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    """Load domain-invariant weights and leave the target adapter newly initialized."""

    excluded_prefixes = (
        "input_domain_adapter.",
        "output_domain_adapter.",
        "link_head.",
        "node_head.",
        "graph_head.",
    )
    destination = target.state_dict()
    selected: dict[str, Tensor] = {}
    for name, value in source_state.items():
        if name.startswith(excluded_prefixes):
            continue
        if name not in destination or destination[name].shape != value.shape:
            raise ValueError(f"LODO shared component is incompatible: {name}")
        selected[name] = value
    required = {name for name in destination if not name.startswith(excluded_prefixes)}
    if set(selected) != required:
        raise ValueError("LODO source checkpoint lacks a complete shared backbone")
    incompatible = target.load_state_dict(selected, strict=False)
    expected_missing = {name for name in destination if name.startswith(excluded_prefixes)}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != expected_missing:
        raise ValueError("LODO load initialized or skipped an undeclared component")
    return tuple(sorted(selected))


__all__ = [
    "DOMAIN_FAMILY_BY_ID",
    "FORMAL_DOMAIN_IDS",
    "FORMAL_SEEDS",
    "LodoIsolationAudit",
    "VariantSelection",
    "assert_lodo_isolation",
    "few_shot_indices",
    "load_lodo_shared_backbone",
    "select_core_variant",
    "select_formal_checkpoints",
]
