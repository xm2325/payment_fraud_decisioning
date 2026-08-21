from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paysim_monitoring import ranked_capacity_metrics


def validation_tail_reference(
    probability: Sequence[float],
    *,
    tail_quantile: float = 0.995,
) -> tuple[float, float]:
    """Fit a score-tail threshold and actual reference tail rate on validation only."""
    if not 0 < tail_quantile < 1:
        raise ValueError("tail_quantile must be in (0, 1)")
    p = np.asarray(probability, dtype=float)
    if len(p) == 0 or np.any(~np.isfinite(p)):
        raise ValueError("probability must be non-empty and finite")
    threshold = float(np.quantile(p, tail_quantile, method="higher"))
    tail_rate = float(np.mean(p >= threshold))
    if tail_rate <= 0:
        raise ValueError("validation tail rate must be positive")
    return threshold, tail_rate


def capacity_from_score_load(
    probability: Sequence[float],
    *,
    tail_threshold: float,
    validation_tail_rate: float,
    baseline_alerts_per_10k: float = 50,
    surge_alerts_per_10k: float = 100,
    trigger_multiplier: float = 1.5,
) -> dict:
    """Choose review capacity from score load only; no labels are used."""
    if validation_tail_rate <= 0:
        raise ValueError("validation_tail_rate must be positive")
    if baseline_alerts_per_10k < 0 or surge_alerts_per_10k < baseline_alerts_per_10k:
        raise ValueError("surge capacity must be >= non-negative baseline capacity")
    if trigger_multiplier <= 0:
        raise ValueError("trigger_multiplier must be positive")
    p = np.asarray(probability, dtype=float)
    if len(p) == 0 or np.any(~np.isfinite(p)):
        raise ValueError("probability must be non-empty and finite")
    tail_rate = float(np.mean(p >= tail_threshold))
    multiplier = float(tail_rate / validation_tail_rate)
    triggered = bool(multiplier >= trigger_multiplier)
    capacity = float(surge_alerts_per_10k if triggered else baseline_alerts_per_10k)
    return {
        "score_tail_rate": tail_rate,
        "score_tail_multiplier_vs_validation": multiplier,
        "surge_triggered": triggered,
        "selected_alerts_per_10k": capacity,
    }


def evaluate_surge_windows(
    step: Sequence[int],
    y: Sequence[int],
    probability: Sequence[float],
    priority_score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    tail_threshold: float,
    validation_tail_rate: float,
    baseline_alerts_per_10k: float = 50,
    surge_alerts_per_10k: float = 100,
    trigger_multiplier: float = 1.5,
    n_windows: int = 3,
) -> pd.DataFrame:
    """Compare frozen baseline capacity with a prospective score-tail surge policy."""
    step_arr = np.asarray(step, dtype=int)
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(probability, dtype=float)
    priority_arr = np.asarray(priority_score, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if not (
        len(step_arr)
        == len(y_arr)
        == len(p_arr)
        == len(priority_arr)
        == len(amount_arr)
        == len(key_arr)
    ):
        raise ValueError("all input arrays must have equal length")
    unique_steps = np.unique(step_arr)
    if len(unique_steps) < n_windows:
        raise ValueError("Need at least one distinct step per window")

    rows: list[dict] = []
    for idx, chunk in enumerate(np.array_split(unique_steps, n_windows), start=1):
        mask = np.isin(step_arr, chunk)
        decision = capacity_from_score_load(
            p_arr[mask],
            tail_threshold=tail_threshold,
            validation_tail_rate=validation_tail_rate,
            baseline_alerts_per_10k=baseline_alerts_per_10k,
            surge_alerts_per_10k=surge_alerts_per_10k,
            trigger_multiplier=trigger_multiplier,
        )
        fixed = ranked_capacity_metrics(
            y_arr[mask], priority_arr[mask], amount_arr[mask], key_arr[mask], baseline_alerts_per_10k
        )
        adaptive = ranked_capacity_metrics(
            y_arr[mask], priority_arr[mask], amount_arr[mask], key_arr[mask], decision["selected_alerts_per_10k"]
        )
        common = {
            "period": f"future_window_{idx}",
            "step_min": int(chunk.min()),
            "step_max": int(chunk.max()),
            "n": int(mask.sum()),
            "fraud_n": int(y_arr[mask].sum()),
            "fraud_rate": float(y_arr[mask].mean()),
            "score_tail_threshold": float(tail_threshold),
            "validation_score_tail_rate": float(validation_tail_rate),
            **decision,
        }
        rows.append({"policy": "fixed_baseline", **common, **fixed})
        rows.append({"policy": "score_tail_surge", **common, **adaptive})
    return pd.DataFrame(rows)
