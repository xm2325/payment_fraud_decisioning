from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


@dataclass(frozen=True)
class ValidationStageSplit:
    calibration_step_min: int
    calibration_step_max: int
    policy_step_min: int
    policy_step_max: int
    policy_cut: int
    calibration_n_steps: int
    policy_n_steps: int


def split_validation_stages(
    step: Sequence[int], *, calibration_fraction: float = 0.5
) -> ValidationStageSplit:
    """Split an ordered validation period into disjoint calibration and policy stages.

    The split depends only on time steps, never labels or model performance. The
    earlier stage fits the probability calibrator; the later stage selects the
    routing policy. Future-test rows are not passed to this function.
    """
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    steps = np.asarray(step, dtype=int)
    if steps.size == 0:
        raise ValueError("step must not be empty")
    unique_steps = np.unique(steps)
    if len(unique_steps) < 4:
        raise ValueError("Need at least four distinct validation steps")

    cut_index = int(np.floor(len(unique_steps) * calibration_fraction))
    cut_index = min(max(cut_index, 1), len(unique_steps) - 1)
    policy_cut = int(unique_steps[cut_index])
    calibration_steps = unique_steps[unique_steps < policy_cut]
    policy_steps = unique_steps[unique_steps >= policy_cut]
    if len(calibration_steps) == 0 or len(policy_steps) == 0:
        raise ValueError("Both calibration and policy stages must be non-empty")

    return ValidationStageSplit(
        calibration_step_min=int(calibration_steps.min()),
        calibration_step_max=int(calibration_steps.max()),
        policy_step_min=int(policy_steps.min()),
        policy_step_max=int(policy_steps.max()),
        policy_cut=policy_cut,
        calibration_n_steps=int(len(calibration_steps)),
        policy_n_steps=int(len(policy_steps)),
    )


def stage_masks(step: Sequence[int], split: ValidationStageSplit) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint masks for the calibration and policy-selection stages."""
    steps = np.asarray(step, dtype=int)
    calibration = steps < split.policy_cut
    policy = steps >= split.policy_cut
    if np.any(calibration & policy) or not np.all(calibration | policy):
        raise AssertionError("Stage masks must be disjoint and exhaustive")
    return calibration, policy


def probability_stage_metrics(
    stage: str,
    y: Sequence[int],
    probability: Sequence[float],
) -> dict:
    """Compact probability diagnostics for a temporal stage."""
    labels = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(labels) != len(p) or len(labels) == 0:
        raise ValueError("y and probability must have equal non-zero length")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and in [0, 1]")
    if len(np.unique(labels)) < 2:
        raise ValueError("Both classes are required for stage diagnostics")
    return {
        "stage": stage,
        "n": int(len(labels)),
        "fraud_n": int(labels.sum()),
        "fraud_rate": float(labels.mean()),
        "mean_predicted_probability": float(p.mean()),
        "brier": float(brier_score_loss(labels, p)),
        "pr_auc": float(average_precision_score(labels, p)),
    }


def split_summary_frame(
    step: Sequence[int], y: Sequence[int], split: ValidationStageSplit
) -> pd.DataFrame:
    """Describe the two validation stages without inspecting future-test data."""
    labels = np.asarray(y, dtype=int)
    calibration, policy = stage_masks(step, split)
    rows = []
    for name, mask in (("calibration", calibration), ("policy_selection", policy)):
        stage_steps = np.asarray(step, dtype=int)[mask]
        rows.append({
            "stage": name,
            "step_min": int(stage_steps.min()),
            "step_max": int(stage_steps.max()),
            "n_steps": int(len(np.unique(stage_steps))),
            "n": int(mask.sum()),
            "fraud_n": int(labels[mask].sum()),
            "fraud_rate": float(labels[mask].mean()),
        })
    return pd.DataFrame(rows)
