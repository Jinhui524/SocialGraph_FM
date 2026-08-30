from __future__ import annotations

import pytest
import torch

from socialgraph_gfm.global_model.calibration import (
    binary_ece,
    calibration_state,
    fit_binary_logit_calibrator,
    select_country_balanced_threshold,
)


def test_temperature_and_bias_fit_validation_logits_without_regressing_nll() -> None:
    labels = torch.tensor([0, 1] * 40, dtype=torch.float32)
    logits = torch.where(labels == 1, torch.full_like(labels, 4.0), torch.full_like(labels, 1.0))
    fit = fit_binary_logit_calibrator(logits, labels, max_iter=60)
    assert fit.sample_count == 80
    assert fit.after_loss <= fit.before_loss
    assert all(not parameter.requires_grad for parameter in fit.calibrator.parameters())
    state = calibration_state(fit.calibrator)
    assert state["schemaVersion"] == "socialgraph-fm.global-model-calibration/1.0"
    assert float(state["temperature"]) > 0
    assert binary_ece(logits, labels, calibrator=fit.calibrator) <= binary_ece(logits, labels)


def test_threshold_selection_equal_weights_countries_and_is_deterministic() -> None:
    logits = {
        "china": torch.tensor([-3.0, -1.0, 0.2, 2.0]),
        "iran": torch.tensor([-2.0, 0.1, 0.4, 3.0, 4.0, 5.0]),
    }
    labels = {
        "china": torch.tensor([0, 0, 1, 1]),
        "iran": torch.tensor([0, 1, 0, 1, 1, 1]),
    }
    first = select_country_balanced_threshold(
        logits, labels, candidates=(0.3, 0.5, 0.7)
    )
    second = select_country_balanced_threshold(
        logits, labels, candidates=(0.7, 0.5, 0.3, 0.5)
    )
    assert first.threshold == second.threshold
    assert first.mean_macro_f1 == sum(first.per_country_macro_f1.values()) / 2
    assert first.candidate_count == 3
    assert set(first.per_country_macro_f1) == {"china", "iran"}


def test_calibration_rejects_single_class_validation_data() -> None:
    with pytest.raises(ValueError, match="both validation classes"):
        fit_binary_logit_calibrator(torch.ones(4), torch.ones(4))
