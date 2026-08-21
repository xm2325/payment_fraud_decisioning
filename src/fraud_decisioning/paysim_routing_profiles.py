from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paysim_monitoring import ranked_capacity_metrics, ranked_capacity_windows


DEFAULT_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PROFILE_ORDER = ("case_first", "balanced", "value_first")


def amount_scale_from_validation(amount: Sequence[float]) -> float:
    """Return a robust positive scale fitted only on validation amounts."""
    arr = np.asarray(amount, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("amount must be finite")
    positive = arr[arr > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def priority_score(
    probability: Sequence[float],
    amount: Sequence[float],
    alpha: float,
    amount_scale: float,
) -> np.ndarray:
    """Rank by calibrated fraud probability times scaled amount**alpha.

    alpha=0 is pure probability ranking; alpha=1 is expected-loss-style
    probability x amount ranking. Intermediate alpha values are pre-specified
    governance trade-offs rather than test-tuned model features.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not np.isfinite(amount_scale) or amount_scale <= 0:
        raise ValueError("amount_scale must be positive and finite")
    p = np.asarray(probability, dtype=float)
    a = np.asarray(amount, dtype=float)
    if len(p) != len(a):
        raise ValueError("probability and amount must have equal length")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must be finite and in [0, 1]")
    if np.any(~np.isfinite(a)):
        raise ValueError("amount must be finite")
    if alpha == 0:
        return p.copy()
    scaled_amount = np.clip(a, 0, None) / amount_scale
    return p * np.power(scaled_amount, alpha)


def alpha_grid_metrics(
    y: Sequence[int],
    probability: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float = 50,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    amount_scale: float | None = None,
) -> pd.DataFrame:
    """Evaluate a pre-specified alpha grid at one exact review capacity."""
    scale = amount_scale if amount_scale is not None else amount_scale_from_validation(amount)
    rows: list[dict] = []
    for alpha in alphas:
        score = priority_score(probability, amount, float(alpha), scale)
        row = ranked_capacity_metrics(y, score, amount, event_key, alerts_per_10k)
        row.update({"alpha": float(alpha), "amount_scale": float(scale)})
        recall = float(row["recall"])
        value_recall = float(row["fraud_value_recall"])
        row["balanced_hmean"] = (
            float(2 * recall * value_recall / (recall + value_recall))
            if recall + value_recall > 0
            else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_profiles(validation_grid: pd.DataFrame) -> pd.DataFrame:
    """Select case/balanced/value profiles using validation metrics only."""
    required = {"alpha", "recall", "precision", "fraud_value_recall", "balanced_hmean", "amount_scale"}
    missing = required.difference(validation_grid.columns)
    if missing:
        raise ValueError(f"validation_grid missing columns: {sorted(missing)}")
    if validation_grid.empty:
        raise ValueError("validation_grid must not be empty")

    specs = {
        "case_first": ["recall", "precision", "fraud_value_recall"],
        "balanced": ["balanced_hmean", "precision", "recall"],
        "value_first": ["fraud_value_recall", "precision", "recall"],
    }
    rows: list[dict] = []
    for profile in PROFILE_ORDER:
        metrics = specs[profile]
        # Final alpha ascending tie-break keeps the less amount-sensitive policy
        # when all validation objectives are equal.
        ordered = validation_grid.sort_values(
            metrics + ["alpha"],
            ascending=[False] * len(metrics) + [True],
            kind="mergesort",
        )
        chosen = ordered.iloc[0].to_dict()
        chosen["profile"] = profile
        chosen["selection_split"] = "validation"
        rows.append(chosen)
    return pd.DataFrame(rows)


def evaluate_selected_profiles(
    selected_profiles: pd.DataFrame,
    y: Sequence[int],
    probability: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    budgets_per_10k: Sequence[float] = (10, 25, 50, 100),
) -> pd.DataFrame:
    """Apply validation-selected alphas to a later split without re-selection."""
    rows: list[dict] = []
    for _, profile_row in selected_profiles.iterrows():
        profile = str(profile_row["profile"])
        alpha = float(profile_row["alpha"])
        scale = float(profile_row["amount_scale"])
        score = priority_score(probability, amount, alpha, scale)
        for budget in budgets_per_10k:
            row = ranked_capacity_metrics(y, score, amount, event_key, float(budget))
            row.update({"profile": profile, "alpha": alpha, "amount_scale": scale})
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_profile_windows(
    selected_profiles: pd.DataFrame,
    step: Sequence[int],
    y: Sequence[int],
    probability: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float = 50,
    n_windows: int = 3,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, profile_row in selected_profiles.iterrows():
        profile = str(profile_row["profile"])
        alpha = float(profile_row["alpha"])
        scale = float(profile_row["amount_scale"])
        score = priority_score(probability, amount, alpha, scale)
        frame = ranked_capacity_windows(
            step, y, score, amount, event_key,
            alerts_per_10k=alerts_per_10k,
            n_windows=n_windows,
        )
        frame.insert(0, "profile", profile)
        frame.insert(1, "alpha", alpha)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
