from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .paysim_policy_promotion import exact_capacity_mask


METRIC_KEYS = ("precision", "fraud_recall", "fraud_value_recall")


def _validate_inputs(step, y, score, amount, event_key):
    arrays = (
        np.asarray(step, dtype=int),
        np.asarray(y, dtype=int),
        np.asarray(score, dtype=float),
        np.asarray(amount, dtype=float),
        np.asarray(event_key, dtype=np.uint64),
    )
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        raise ValueError("step, y, score, amount and event_key must have equal non-zero length")
    if np.any(~np.isfinite(arrays[2])):
        raise ValueError("score must be finite")
    if np.any(~np.isfinite(arrays[3])) or np.any(arrays[3] < 0):
        raise ValueError("amount must be finite and non-negative")
    if np.any((arrays[1] != 0) & (arrays[1] != 1)):
        raise ValueError("y must contain only 0/1 labels")
    return arrays


def _metrics(y: np.ndarray, amount: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    fraud = y == 1
    alerts = int(selected.sum())
    fraud_n = int(fraud.sum())
    fraud_alerts = int((selected & fraud).sum())
    fraud_value_total = float(amount[fraud].sum())
    fraud_value_captured = float(amount[selected & fraud].sum())
    return {
        "alerts": alerts,
        "fraud_alerts": fraud_alerts,
        "precision": float(fraud_alerts / alerts) if alerts else math.nan,
        "fraud_recall": float(fraud_alerts / fraud_n) if fraud_n else math.nan,
        "fraud_value_recall": (
            float(fraud_value_captured / fraud_value_total) if fraud_value_total > 0 else math.nan
        ),
        "fraud_value_captured": fraud_value_captured,
    }


def online_accrual_capacity_mask(
    step: Sequence[int],
    score: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Allocate a cumulative alert budget without comparing against future steps.

    At each observed step, the cumulative entitlement is
    floor(alerts_per_10k * cumulative_transactions / 10_000). Only the current
    step's scores may compete for newly available slots. Fractional entitlement
    carries forward through the cumulative floor. Equal scores are resolved by
    the same stable non-label event key used by the batch exact-capacity rule.

    This is a step-level micro-batch contract. It removes cross-step score
    look-ahead but does not claim transaction-level ordering inside one PaySim
    step, because the dataset does not provide a finer ordering contract here.
    """
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    step_arr = np.asarray(step, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if len(step_arr) == 0 or len(step_arr) != len(score_arr) or len(step_arr) != len(key_arr):
        raise ValueError("step, score and event_key must have equal non-zero length")
    if np.any(~np.isfinite(score_arr)):
        raise ValueError("score must be finite")

    selected = np.zeros(len(step_arr), dtype=bool)
    cumulative_n = 0
    selected_total = 0
    schedule: list[dict] = []

    for value in np.unique(step_arr):
        idx = np.flatnonzero(step_arr == value)
        cumulative_n += len(idx)
        entitlement = min(
            cumulative_n,
            int(np.floor(float(alerts_per_10k) * cumulative_n / 10_000)),
        )
        available = max(0, entitlement - selected_total)
        take = min(available, len(idx))
        if take:
            local_order = np.lexsort((key_arr[idx], -score_arr[idx]))
            chosen = idx[local_order[:take]]
            selected[chosen] = True
            selected_total += take

        schedule.append(
            {
                "step": int(value),
                "transactions": int(len(idx)),
                "cumulative_transactions": int(cumulative_n),
                "cumulative_entitlement": int(entitlement),
                "new_slots": int(available),
                "selected_this_step": int(take),
                "selected_cumulative": int(selected_total),
            }
        )

    final_entitlement = min(
        len(step_arr),
        int(np.floor(float(alerts_per_10k) * len(step_arr) / 10_000)),
    )
    if selected_total != final_entitlement:
        raise AssertionError("Online accrual must consume the final cumulative entitlement")
    return selected, pd.DataFrame(schedule)


def online_capacity_metrics(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[dict, pd.DataFrame]:
    step_arr, y_arr, score_arr, amount_arr, key_arr = _validate_inputs(
        step, y, score, amount, event_key
    )
    selected, schedule = online_accrual_capacity_mask(
        step_arr, score_arr, key_arr, alerts_per_10k=alerts_per_10k
    )
    metrics = _metrics(y_arr, amount_arr, selected)
    metrics.update(
        {
            "target_alerts_per_10k": float(alerts_per_10k),
            "actual_alerts_per_10k": float(selected.mean() * 10_000),
            "routing_contract": "stepwise_cumulative_accrual",
        }
    )
    return metrics, schedule


def batch_vs_online_capacity(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[dict, pd.DataFrame]:
    """Compare retrospective whole-window top-k with causal stepwise capacity."""
    step_arr, y_arr, score_arr, amount_arr, key_arr = _validate_inputs(
        step, y, score, amount, event_key
    )
    batch_selected = exact_capacity_mask(score_arr, key_arr, alerts_per_10k)
    online_selected, schedule = online_accrual_capacity_mask(
        step_arr, score_arr, key_arr, alerts_per_10k=alerts_per_10k
    )
    batch = _metrics(y_arr, amount_arr, batch_selected)
    online = _metrics(y_arr, amount_arr, online_selected)

    overlap = int((batch_selected & online_selected).sum())
    batch_alerts = int(batch_selected.sum())
    online_alerts = int(online_selected.sum())
    union = int((batch_selected | online_selected).sum())
    batch_only = batch_selected & ~online_selected
    online_only = online_selected & ~batch_selected
    fraud = y_arr == 1

    row: dict[str, float | int | str] = {
        "target_alerts_per_10k": float(alerts_per_10k),
        "batch_alerts": batch_alerts,
        "online_alerts": online_alerts,
        "queue_overlap": overlap,
        "queue_overlap_rate": float(overlap / batch_alerts) if batch_alerts else math.nan,
        "queue_jaccard": float(overlap / union) if union else 1.0,
        "replacement_count": int(online_only.sum()),
        "replacement_rate": float(online_only.sum() / batch_alerts) if batch_alerts else math.nan,
        "batch_only_fraud": int((batch_only & fraud).sum()),
        "online_only_fraud": int((online_only & fraud).sum()),
        "batch_only_fraud_value": float(amount_arr[batch_only & fraud].sum()),
        "online_only_fraud_value": float(amount_arr[online_only & fraud].sum()),
        "incremental_fraud_cases_from_online_swaps": int(
            (online_only & fraud).sum() - (batch_only & fraud).sum()
        ),
        "incremental_fraud_value_from_online_swaps": float(
            amount_arr[online_only & fraud].sum() - amount_arr[batch_only & fraud].sum()
        ),
    }
    for metric in METRIC_KEYS:
        row[f"batch_{metric}"] = float(batch[metric])
        row[f"online_{metric}"] = float(online[metric])
        row[f"delta_{metric}"] = float(online[metric] - batch[metric])

    if batch_alerts != online_alerts:
        raise AssertionError("Batch and online rules must consume the same final total capacity")
    return row, schedule
