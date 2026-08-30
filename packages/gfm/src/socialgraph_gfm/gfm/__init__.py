"""Public training-kernel API for SocialGraph-FM Core.

The package is also the parent namespace for corpus-only commands.  Public
symbols are therefore resolved lazily so a license/fetch/check process does
not import the 500+ MiB Torch/PyG runtime merely by importing
``socialgraph_gfm.gfm.corpus`` or the workflow dispatcher.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CalibrationMetrics": (".evaluation", "CalibrationMetrics"),
    "DomainTransferResult": (".evaluation", "DomainTransferResult"),
    "LeaveOneDomainOutSummary": (".evaluation", "LeaveOneDomainOutSummary"),
    "RankingMetrics": (".evaluation", "RankingMetrics"),
    "evaluate_lodo": (".evaluation", "evaluate_lodo"),
    "expected_calibration_error": (".evaluation", "expected_calibration_error"),
    "ranking_metrics": (".evaluation", "ranking_metrics"),
    "SocialGraphFMCore": (".model", "SocialGraphFMCore"),
    "FixedObjectiveWeights": (".objectives", "FixedObjectiveWeights"),
    "LossBundle": (".objectives", "LossBundle"),
    "OBJECTIVE_WEIGHTS": (".objectives", "OBJECTIVE_WEIGHTS"),
    "compute_fixed_multiloss": (".objectives", "compute_fixed_multiloss"),
    "CausalExactNegativeSampler": (".sampling", "CausalExactNegativeSampler"),
    "CausalMixedNegativeSampler": (".sampling", "CausalMixedNegativeSampler"),
    "MixedNegativeSample": (".sampling", "MixedNegativeSample"),
    "RoundRobinDomainScheduler": (".sampling", "RoundRobinDomainScheduler"),
    "ProductAdaptBatch": (".product_training", "ProductAdaptBatch"),
    "ProductPredictionReport": (".product_training", "ProductPredictionReport"),
    "ProductResumeState": (".product_training", "ProductResumeState"),
    "ProductProgressCallback": (".product_training", "ProductProgressCallback"),
    "ProductTaskModule": (".product_training", "ProductTaskModule"),
    "ProductTrainingConfig": (".product_training", "ProductTrainingConfig"),
    "ProductTrainingResult": (".product_training", "ProductTrainingResult"),
    "SampleProvenance": (".product_training", "SampleProvenance"),
    "calibration_by_stratum": (".product_training", "calibration_by_stratum"),
    "evaluate_product_predictions": (
        ".product_training",
        "evaluate_product_predictions",
    ),
    "train_product_steps": (".product_training", "train_product_steps"),
    "CoreTrainer": (".trainer", "CoreTrainer"),
    "CoreTrainerConfig": (".trainer", "CoreTrainerConfig"),
    "TrainingEpochResult": (".trainer", "TrainingEpochResult"),
    "LodoIsolationAudit": (".transfer_workflow", "LodoIsolationAudit"),
    "VariantSelection": (".transfer_workflow", "VariantSelection"),
    "assert_lodo_isolation": (".transfer_workflow", "assert_lodo_isolation"),
    "few_shot_indices": (".transfer_workflow", "few_shot_indices"),
    "load_lodo_shared_backbone": (
        ".transfer_workflow",
        "load_lodo_shared_backbone",
    ),
    "select_core_variant": (".transfer_workflow", "select_core_variant"),
    "select_formal_checkpoints": (
        ".transfer_workflow",
        "select_formal_checkpoints",
    ),
    "CoreBatch": (".types", "CoreBatch"),
    "CoreModelConfig": (".types", "CoreModelConfig"),
    "CoreOutput": (".types", "CoreOutput"),
    "CoreSampleProvenance": (".types", "CoreSampleProvenance"),
}

__all__ = sorted(_EXPORTS)


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
