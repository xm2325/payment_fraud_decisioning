import numpy as np
import pytest

from fraud_decisioning.paysim_rolling_refresh import (
    build_rolling_cycles,
    cycle_contract_frame,
    validate_cycle_sequence,
)


def test_build_rolling_cycles_is_expanding_and_future_safe():
    cycles = build_rolling_cycles(
        np.arange(1, 31),
        initial_test_step=21,
        calibration_n_steps=4,
        policy_n_steps=5,
        test_window_n_steps=5,
    )
    assert len(cycles) == 2

    first, second = cycles
    assert (first.train_step_min, first.train_step_max) == (1, 11)
    assert (first.calibration_step_min, first.calibration_step_max) == (12, 15)
    assert (first.policy_step_min, first.policy_step_max) == (16, 20)
    assert (first.test_step_min, first.test_step_max) == (21, 25)

    assert (second.train_step_min, second.train_step_max) == (1, 16)
    assert (second.calibration_step_min, second.calibration_step_max) == (17, 20)
    assert (second.policy_step_min, second.policy_step_max) == (21, 25)
    assert (second.test_step_min, second.test_step_max) == (26, 30)

    validate_cycle_sequence(cycles)
    frame = cycle_contract_frame(cycles)
    assert frame.test_n_steps.tolist() == [5, 5]
    assert frame.train_n_steps.tolist() == [11, 16]


def test_last_test_window_can_be_shorter_without_leakage():
    cycles = build_rolling_cycles(
        np.arange(1, 29),
        initial_test_step=21,
        calibration_n_steps=4,
        policy_n_steps=5,
        test_window_n_steps=5,
    )
    assert cycles[-1].test_step_min == 26
    assert cycles[-1].test_step_max == 28
    assert cycles[-1].test_n_steps == 3
    validate_cycle_sequence(cycles)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_test_step": 99, "calibration_n_steps": 4, "policy_n_steps": 5},
        {"initial_test_step": 8, "calibration_n_steps": 4, "policy_n_steps": 5},
        {"initial_test_step": 21, "calibration_n_steps": 0, "policy_n_steps": 5},
    ],
)
def test_invalid_rolling_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_rolling_cycles(np.arange(1, 31), **kwargs)
