from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


@dataclass(frozen=True)
class FutureWindow:
    name: str
    step_min: int
    step_max: int


def contiguous_future_windows(step: Sequence[int], n_windows: int = 3) -> list[FutureWindow]:
    """Split future steps into contiguous, deterministic windows."""
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    steps = np.asarray(step, dtype=int)
    unique_steps = np.unique(steps)
    if len(unique_steps) < n_windows:
        raise ValueError("Need at least one distinct future step per window")
    chunks = [chunk for chunk in np.array_split(unique_steps, n_windows) if len(chunk)]
    return [
        FutureWindow(
            name=f"future_window_{idx}",
            step_min=int(chunk.min()),
            step_max=int(chunk.max()),
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]


def matured_incremental_mask(
    step: Sequence[int],
    *,
    window_start_step: int,
    maturity_lag_steps: int,
    initial_calibration_step_max: int,
) -> tuple[np.ndarray, int]:
    """Return post-initial-calibration labels available before a future window.

    The initial approved calibration set is assumed to exist already. The lag
    constrains only *new* labels considered for later refreshes. A label from
    transaction step s is available before scoring step t only when
    s + maturity_lag_steps < t, so the maximum eligible step is t-lag-1.
    """
    if maturity_lag_steps < 0:
        raise ValueError("maturity_lag_steps must be non-negative")
    steps = np.asarray(step, dtype=int)
    cutoff = int(window_start_step - maturity_lag_steps - 1)
    mask = (steps > int(initial_calibration_step_max)) & (steps <= cutoff)
    return mask, cutoff


def expanding_recalibration_mask(
    step: Sequence[int],
    *,
    window_start_step: int,
    maturity_lag_steps: int,
    initial_calibration_step_min: int,
    initial_calibration_step_max: int,
) -> tuple[np.ndarray, int]:
    """Return initial approved calibration rows plus all matured later labels."""
    steps = np.asarray(step, dtype=int)
    base = (
        (steps >= int(initial_calibration_step_min))
        & (steps <= int(initial_calibration_step_max))
    )
    incremental, cutoff = matured_incremental_mask(
        steps,
        window_start_step=window_start_step,
        maturity_lag_steps=maturity_lag_steps,
        initial_calibration_step_max=initial_calibration_step_max,
    )
    return base | incremental, cutoff


def recalibration_metric_row(
    *,
    method: str,
    window: FutureWindow,
    y: Sequence[int],
    probability: Sequence[float],
    calibration_n: int,
    calibration_max_step: int,
    maturity_lag_steps: int | None,
) -> dict:
    labels = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(labels) != len(p) or len(labels) == 0:
        raise ValueError("y and probability must have equal non-zero length")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and in [0, 1]")
    if len(np.unique(labels)) < 2:
        raise ValueError("Both classes are required for recalibration diagnostics")
    prevalence = float(labels.mean())
    mean_probability = float(p.mean())
    return {
        "method": method,
        "window": window.name,
        "step_min": window.step_min,
        "step_max": window.step_max,
        "n": int(len(labels)),
        "fraud_n": int(labels.sum()),
        "fraud_rate": prevalence,
        "mean_predicted_probability": mean_probability,
        "mean_to_observed_ratio": float(mean_probability / prevalence) if prevalence > 0 else np.nan,
        "brier": float(brier_score_loss(labels, p)),
        "pr_auc": float(average_precision_score(labels, p)),
        "calibration_n": int(calibration_n),
        "calibration_max_step": int(calibration_max_step),
        "maturity_lag_steps": None if maturity_lag_steps is None else int(maturity_lag_steps),
    }


def method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise calibration error across future windows without selecting on future labels."""
    required = {"method", "brier", "mean_to_observed_ratio"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    rows = []
    for method, group in frame.groupby("method", sort=False):
        rows.append({
            "method": method,
            "mean_brier": float(group.brier.mean()),
            "worst_brier": float(group.brier.max()),
            "mean_abs_log_risk_ratio": float(
                np.mean(np.abs(np.log(np.clip(group.mean_to_observed_ratio.to_numpy(float), 1e-12, None))))
            ),
            "n_windows": int(len(group)),
        })
    return pd.DataFrame(rows)
