from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingCycle:
    cycle: int
    train_step_min: int
    train_step_max: int
    calibration_step_min: int
    calibration_step_max: int
    policy_step_min: int
    policy_step_max: int
    test_step_min: int
    test_step_max: int
    train_n_steps: int
    calibration_n_steps: int
    policy_n_steps: int
    test_n_steps: int


def build_rolling_cycles(
    step: Sequence[int],
    *,
    initial_test_step: int,
    calibration_n_steps: int,
    policy_n_steps: int,
    test_window_n_steps: int = 50,
) -> list[RollingCycle]:
    """Build deterministic expanding-history, rolling-origin evaluation cycles.

    Every cycle uses all distinct steps before its calibration window for model
    training, then a fixed-length calibration window, then a fixed-length
    routing-policy window, then the next test window. Boundaries depend only on
    time steps and the declared window lengths, never labels or performance.
    """
    if calibration_n_steps < 1 or policy_n_steps < 1 or test_window_n_steps < 1:
        raise ValueError("All window lengths must be positive")

    unique_steps = np.unique(np.asarray(step, dtype=int))
    if len(unique_steps) == 0:
        raise ValueError("step must not be empty")

    matches = np.flatnonzero(unique_steps == int(initial_test_step))
    if len(matches) != 1:
        raise ValueError("initial_test_step must be one observed step")
    initial_test_index = int(matches[0])

    history_needed = calibration_n_steps + policy_n_steps
    if initial_test_index <= history_needed:
        raise ValueError("Not enough history before the initial test step")

    cycles: list[RollingCycle] = []
    for cycle_id, test_start_index in enumerate(
        range(initial_test_index, len(unique_steps), test_window_n_steps), start=1
    ):
        policy_start_index = test_start_index - policy_n_steps
        calibration_start_index = policy_start_index - calibration_n_steps
        if calibration_start_index <= 0:
            raise ValueError("Each cycle requires at least one training step")

        test_end_index = min(test_start_index + test_window_n_steps, len(unique_steps))
        train_steps = unique_steps[:calibration_start_index]
        calibration_steps = unique_steps[calibration_start_index:policy_start_index]
        policy_steps = unique_steps[policy_start_index:test_start_index]
        test_steps = unique_steps[test_start_index:test_end_index]

        if not (
            train_steps[-1] < calibration_steps[0]
            <= calibration_steps[-1] < policy_steps[0]
            <= policy_steps[-1] < test_steps[0]
        ):
            raise AssertionError("Rolling cycle stages must be strictly ordered")

        cycles.append(
            RollingCycle(
                cycle=cycle_id,
                train_step_min=int(train_steps[0]),
                train_step_max=int(train_steps[-1]),
                calibration_step_min=int(calibration_steps[0]),
                calibration_step_max=int(calibration_steps[-1]),
                policy_step_min=int(policy_steps[0]),
                policy_step_max=int(policy_steps[-1]),
                test_step_min=int(test_steps[0]),
                test_step_max=int(test_steps[-1]),
                train_n_steps=int(len(train_steps)),
                calibration_n_steps=int(len(calibration_steps)),
                policy_n_steps=int(len(policy_steps)),
                test_n_steps=int(len(test_steps)),
            )
        )
    return cycles


def cycle_contract_frame(cycles: Sequence[RollingCycle]) -> pd.DataFrame:
    """Return one row per cycle for audit and documentation."""
    if not cycles:
        raise ValueError("cycles must not be empty")
    return pd.DataFrame([cycle.__dict__ for cycle in cycles])


def validate_cycle_sequence(cycles: Sequence[RollingCycle]) -> None:
    """Fail if a declared rolling schedule can leak future steps."""
    if not cycles:
        raise ValueError("cycles must not be empty")
    previous_test_min = None
    previous_train_max = None
    for cycle in cycles:
        if not (
            cycle.train_step_max < cycle.calibration_step_min
            <= cycle.calibration_step_max < cycle.policy_step_min
            <= cycle.policy_step_max < cycle.test_step_min
            <= cycle.test_step_max
        ):
            raise ValueError(f"Cycle {cycle.cycle} violates temporal ordering")
        if previous_test_min is not None and cycle.test_step_min <= previous_test_min:
            raise ValueError("Test origins must move strictly forward")
        if previous_train_max is not None and cycle.train_step_max <= previous_train_max:
            raise ValueError("Training history must expand between cycles")
        previous_test_min = cycle.test_step_min
        previous_train_max = cycle.train_step_max
