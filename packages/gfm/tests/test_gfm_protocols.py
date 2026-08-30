from datetime import UTC, datetime

import pytest

from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.protocols import (
    COLLABORATION_WINDOWS,
    DOMAIN_FAMILIES,
    NEWCOMER_PROTOCOL,
    assert_cutoff_safe,
    collaboration_stratum,
    lodo_training_families,
)


def test_collaboration_windows_freeze_train_validation_test_and_shadow():
    assert [(item.role, item.cutoff_year, item.target_year) for item in COLLABORATION_WINDOWS] == [
        ("train", 2017, 2018),
        ("train", 2018, 2019),
        ("train", 2019, 2020),
        ("train", 2020, 2021),
        ("train", 2021, 2022),
        ("validation", 2022, 2023),
        ("test", 2023, 2024),
        ("shadow", 2024, 2025),
    ]
    assert NEWCOMER_PROTOCOL.observation_days == 90
    assert NEWCOMER_PROTOCOL.horizon_days == 365


def test_lodo_uses_domain_families_not_multiple_academic_datasets():
    for held_out in DOMAIN_FAMILIES:
        training = lodo_training_families(held_out)
        assert held_out not in training
        assert len(training) == 2


def test_cutoff_and_first_time_strata_are_fail_closed():
    cutoff = datetime(2022, 12, 31, tzinfo=UTC)
    assert_cutoff_safe([datetime(2022, 1, 1, tzinfo=UTC), cutoff], cutoff=cutoff)
    with pytest.raises(ContractViolation, match="future"):
        assert_cutoff_safe([datetime(2023, 1, 1, tzinfo=UTC)], cutoff=cutoff)
    history = {(1, 2)}
    assert collaboration_stratum(2, 1, history) == "repeated"
    assert collaboration_stratum(1, 3, history) == "first_time"
