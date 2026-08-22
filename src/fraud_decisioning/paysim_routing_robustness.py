from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paysim_routing_profiles import (
    DEFAULT_ALPHA_GRID,
    alpha_grid_metrics,
    amount_scale_from_validation,
)

ROBUST_PROFILE_ORDER = ("case_first", "balanced", "value_first")


def validation_window_alpha_grid(
    step: Sequence[int],
    y: Sequence[int],
    probability: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float = 50,
    n_windows: int = 3,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> pd.DataFrame:
    """Evaluate a fixed alpha grid in contiguous validation windows.

    The amount scale is fitted once on the whole validation period, then frozen
    across windows. No future-test information enters this audit.
    """
    step_arr = np.asarray(step, dtype=int)
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probability, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if not (len(step_arr) == len(y_arr) == len(p_arr) == len(amount_arr) == len(key_arr)):
        raise ValueError("step, y, probability, amount and event_key must have equal length")
    unique_steps = np.unique(step_arr)
    if len(unique_steps) < n_windows:
        raise ValueError("Need at least one distinct validation step per window")

    scale = amount_scale_from_validation(amount_arr)
    chunks = [chunk for chunk in np.array_split(unique_steps, n_windows) if len(chunk)]
    rows: list[pd.DataFrame] = []
    for idx, chunk in enumerate(chunks, start=1):
        mask = np.isin(step_arr, chunk)
        frame = alpha_grid_metrics(
            y_arr[mask],
            p_arr[mask],
            amount_arr[mask],
            key_arr[mask],
            alerts_per_10k=alerts_per_10k,
            alphas=alphas,
            amount_scale=scale,
        )
        frame.insert(0, "window", f"validation_window_{idx}")
        frame.insert(1, "step_min", int(chunk.min()))
        frame.insert(2, "step_max", int(chunk.max()))
        frame.insert(3, "n", int(mask.sum()))
        frame.insert(4, "fraud_n", int(y_arr[mask].sum()))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def robustness_summary(window_grid: pd.DataFrame) -> pd.DataFrame:
    """Summarise worst-window and mean validation performance by alpha."""
    required = {
        "alpha", "amount_scale", "recall", "precision",
        "fraud_value_recall", "balanced_hmean",
    }
    missing = required.difference(window_grid.columns)
    if missing:
        raise ValueError(f"window_grid missing columns: {sorted(missing)}")
    if window_grid.empty:
        raise ValueError("window_grid must not be empty")

    rows: list[dict] = []
    for alpha, group in window_grid.groupby("alpha", sort=True):
        rows.append({
            "alpha": float(alpha),
            "amount_scale": float(group.amount_scale.iloc[0]),
            "min_recall": float(group.recall.min()),
            "mean_recall": float(group.recall.mean()),
            "min_precision": float(group.precision.min()),
            "mean_precision": float(group.precision.mean()),
            "min_fraud_value_recall": float(group.fraud_value_recall.min()),
            "mean_fraud_value_recall": float(group.fraud_value_recall.mean()),
            "min_balanced_hmean": float(group.balanced_hmean.min()),
            "mean_balanced_hmean": float(group.balanced_hmean.mean()),
            "recall_range": float(group.recall.max() - group.recall.min()),
            "value_recall_range": float(group.fraud_value_recall.max() - group.fraud_value_recall.min()),
        })
    return pd.DataFrame(rows)


def select_robust_profiles(summary: pd.DataFrame) -> pd.DataFrame:
    """Select routing profiles using validation worst-window metrics only."""
    if summary.empty:
        raise ValueError("summary must not be empty")
    specs = {
        "case_first": ["min_recall", "mean_recall", "mean_precision", "mean_fraud_value_recall"],
        "balanced": ["min_balanced_hmean", "mean_balanced_hmean", "mean_precision"],
        "value_first": [
            "min_fraud_value_recall", "mean_fraud_value_recall", "mean_precision", "mean_recall"
        ],
    }
    rows: list[dict] = []
    for profile in ROBUST_PROFILE_ORDER:
        metrics = specs[profile]
        ordered = summary.sort_values(
            metrics + ["alpha"],
            ascending=[False] * len(metrics) + [True],
            kind="mergesort",
        )
        chosen = ordered.iloc[0].to_dict()
        chosen["profile"] = profile
        chosen["selection_split"] = "validation_windows_only"
        rows.append(chosen)
    return pd.DataFrame(rows)
