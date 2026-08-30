import torch

from socialgraph_gfm.gfm.calibration import fit_temperature


def test_temperature_scaling_uses_only_supplied_validation_values():
    logits = torch.tensor([4.0, 2.0, -2.0, -4.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    before = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    scaler = fit_temperature(logits, labels)
    after = torch.nn.functional.binary_cross_entropy_with_logits(scaler(logits), labels)
    assert torch.isfinite(scaler.temperature)
    assert after <= before + 1e-6
