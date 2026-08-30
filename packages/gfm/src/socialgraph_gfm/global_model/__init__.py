"""Lazy public API for SocialGraph-FM Global data, model, routing, and calibration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "BinaryLogitCalibrator": (".calibration", "BinaryLogitCalibrator"),
    "SparseTop2Router": (".model", "SparseTop2Router"),
    "COUNTRY_IDS": (".contracts", "COUNTRY_IDS"),
    "CalibrationFit": (".calibration", "CalibrationFit"),
    "GRAPH_STATS_DIM": (".model", "GRAPH_STATS_DIM"),
    "GRAPH_STAT_NAMES": (".contracts", "GRAPH_STAT_NAMES"),
    "GlobalCrossModalBackbone": (".model", "GlobalCrossModalBackbone"),
    "GlobalModel": (".model", "GlobalModel"),
    "GlobalModelConfig": (".model", "GlobalModelConfig"),
    "GlobalOutput": (".model", "GlobalOutput"),
    "GlobalArrayDescriptor": (".contracts", "GlobalArrayDescriptor"),
    "GlobalAdjacencyCSR": (".corpus", "GlobalAdjacencyCSR"),
    "GlobalCorpusEntry": (".contracts", "GlobalCorpusEntry"),
    "GlobalCorpusIndex": (".corpus", "GlobalCorpusIndex"),
    "GlobalCorpusManifest": (".contracts", "GlobalCorpusManifest"),
    "GlobalCountryCorpus": (".corpus", "GlobalCountryCorpus"),
    "GlobalCountryManifest": (".contracts", "GlobalCountryManifest"),
    "GlobalRelationCSR": (".corpus", "GlobalRelationCSR"),
    "GlobalSplit": (".corpus", "GlobalSplit"),
    "GlobalSplitDescriptor": (".contracts", "GlobalSplitDescriptor"),
    "TRACE_NAMES": (".contracts", "TRACE_NAMES"),
    "TRACE_ARRAY_TOKENS": (".contracts", "TRACE_ARRAY_TOKENS"),
    "ThresholdSelection": (".calibration", "ThresholdSelection"),
    "atomic_write_contract": (".contracts", "atomic_write_contract"),
    "binary_ece": (".calibration", "binary_ece"),
    "calibration_state": (".calibration", "calibration_state"),
    "degree_bucket_one_hot": (".model", "degree_bucket_one_hot"),
    "fit_binary_logit_calibrator": (".calibration", "fit_binary_logit_calibrator"),
    "load_corpus_index": (".corpus", "load_corpus_index"),
    "load_country_corpus": (".corpus", "load_country_corpus"),
    "read_corpus_manifest": (".contracts", "read_corpus_manifest"),
    "read_country_manifest": (".contracts", "read_country_manifest"),
    "router_load_balancing_loss": (".model", "router_load_balancing_loss"),
    "run_checkpoint_forward_smoke": (".forward_smoke", "run_checkpoint_forward_smoke"),
    "select_country_balanced_threshold": (
        ".calibration",
        "select_country_balanced_threshold",
    ),
}

__all__ = [
    "COUNTRY_IDS",
    "GRAPH_STATS_DIM",
    "GRAPH_STAT_NAMES",
    "TRACE_ARRAY_TOKENS",
    "TRACE_NAMES",
    "BinaryLogitCalibrator",
    "CalibrationFit",
    "GlobalAdjacencyCSR",
    "GlobalArrayDescriptor",
    "GlobalCorpusEntry",
    "GlobalCorpusIndex",
    "GlobalCorpusManifest",
    "GlobalCountryCorpus",
    "GlobalCountryManifest",
    "GlobalCrossModalBackbone",
    "GlobalModel",
    "GlobalModelConfig",
    "GlobalOutput",
    "GlobalRelationCSR",
    "GlobalSplit",
    "GlobalSplitDescriptor",
    "SparseTop2Router",
    "ThresholdSelection",
    "atomic_write_contract",
    "binary_ece",
    "calibration_state",
    "degree_bucket_one_hot",
    "fit_binary_logit_calibrator",
    "load_corpus_index",
    "load_country_corpus",
    "read_corpus_manifest",
    "read_country_manifest",
    "router_load_balancing_loss",
    "run_checkpoint_forward_smoke",
    "select_country_balanced_threshold",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
