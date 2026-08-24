from __future__ import annotations

import heapq
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


def _validate_routing_inputs(step, score, event_key):
    step_arr = np.asarray(step, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if len(step_arr) == 0 or len(step_arr) != len(score_arr) or len(step_arr) != len(key_arr):
        raise ValueError("step, score and event_key must have equal non-zero length")
    if np.any(~np.isfinite(score_arr)):
        raise ValueError("score must be finite")
    return step_arr, score_arr, key_arr


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


def _final_entitlement(n: int, alerts_per_10k: float) -> int:
    return min(n, int(np.floor(float(alerts_per_10k) * n / 10_000)))


def online_accrual_capacity_mask(
    step: Sequence[int],
    score: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Allocate capacity step by step using only the current step's arrivals.

    At each observed step, cumulative entitlement is
    floor(alerts_per_10k * cumulative_transactions / 10_000). Only the current
    step's scores may compete for newly available slots. Fractional entitlement
    carries forward through the cumulative floor. Equal scores use the same
    stable non-label event key as the retrospective batch rule.

    This is a strict low-latency micro-batch contract. It removes cross-step
    score look-ahead but does not claim a finer ordering inside one PaySim step.
    """
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    step_arr, score_arr, key_arr = _validate_routing_inputs(step, score, event_key)

    selected = np.zeros(len(step_arr), dtype=bool)
    cumulative_n = 0
    selected_total = 0
    schedule: list[dict] = []

    for value in np.unique(step_arr):
        idx = np.flatnonzero(step_arr == value)
        cumulative_n += len(idx)
        entitlement = _final_entitlement(cumulative_n, alerts_per_10k)
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
                "pending_after_selection": 0,
            }
        )

    if selected_total != _final_entitlement(len(step_arr), alerts_per_10k):
        raise AssertionError("Stepwise routing must consume final cumulative entitlement")
    return selected, pd.DataFrame(schedule)


def online_backlog_capacity_mask(
    step: Sequence[int],
    score: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Allocate new capacity from all seen-but-unreviewed transactions.

    New arrivals enter a priority backlog. At each step, only capacity earned by
    transactions observed up to that step is available, and the best scores in
    the seen-so-far backlog receive those slots. No later-step score can affect
    an earlier decision. Unlike the strict current-step rule, rejected past
    transactions can remain eligible for later analyst capacity.
    """
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    step_arr, score_arr, key_arr = _validate_routing_inputs(step, score, event_key)

    selected = np.zeros(len(step_arr), dtype=bool)
    review_step = np.full(len(step_arr), -1, dtype=int)
    pending: list[tuple[float, int, int]] = []
    cumulative_n = 0
    selected_total = 0
    schedule: list[dict] = []

    for value in np.unique(step_arr):
        idx = np.flatnonzero(step_arr == value)
        for pos in idx:
            heapq.heappush(
                pending,
                (-float(score_arr[pos]), int(key_arr[pos]), int(pos)),
            )

        cumulative_n += len(idx)
        entitlement = _final_entitlement(cumulative_n, alerts_per_10k)
        available = max(0, entitlement - selected_total)
        take = min(available, len(pending))
        pending_before = len(pending)

        for _ in range(take):
            _, _, pos = heapq.heappop(pending)
            selected[pos] = True
            review_step[pos] = int(value)
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
                "pending_before_selection": int(pending_before),
                "pending_after_selection": int(len(pending)),
            }
        )

    if selected_total != _final_entitlement(len(step_arr), alerts_per_10k):
        raise AssertionError("Backlog routing must consume final cumulative entitlement")
    if np.any(review_step[selected] < step_arr[selected]):
        raise AssertionError("A transaction cannot be reviewed before it arrives")
    return selected, review_step, pd.DataFrame(schedule)


def _queue_comparison(
    step_arr: np.ndarray,
    y_arr: np.ndarray,
    amount_arr: np.ndarray,
    batch_selected: np.ndarray,
    online_selected: np.ndarray,
    *,
    alerts_per_10k: float,
    contract: str,
    review_step: np.ndarray | None = None,
) -> dict:
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
        "routing_contract": contract,
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

    if review_step is not None and online_alerts:
        delay = review_step[online_selected] - step_arr[online_selected]
        row.update(
            {
                "review_delay_mean_steps": float(np.mean(delay)),
                "review_delay_p50_steps": float(np.quantile(delay, 0.50)),
                "review_delay_p90_steps": float(np.quantile(delay, 0.90)),
                "review_delay_max_steps": int(np.max(delay)),
            }
        )
        fraud_delay = delay[y_arr[online_selected] == 1]
        row["fraud_review_delay_mean_steps"] = (
            float(np.mean(fraud_delay)) if len(fraud_delay) else math.nan
        )
        row["fraud_review_delay_p90_steps"] = (
            float(np.quantile(fraud_delay, 0.90)) if len(fraud_delay) else math.nan
        )
    else:
        row.update(
            {
                "review_delay_mean_steps": 0.0,
                "review_delay_p50_steps": 0.0,
                "review_delay_p90_steps": 0.0,
                "review_delay_max_steps": 0,
                "fraud_review_delay_mean_steps": 0.0,
                "fraud_review_delay_p90_steps": 0.0,
            }
        )

    if batch_alerts != online_alerts:
        raise AssertionError("Batch and causal rules must consume the same final total capacity")
    return row


def batch_vs_online_capacity(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[dict, pd.DataFrame]:
    """Compare whole-window hindsight top-k with strict current-step routing."""
    step_arr, y_arr, score_arr, amount_arr, key_arr = _validate_inputs(
        step, y, score, amount, event_key
    )
    batch_selected = exact_capacity_mask(score_arr, key_arr, alerts_per_10k)
    online_selected, schedule = online_accrual_capacity_mask(
        step_arr, score_arr, key_arr, alerts_per_10k=alerts_per_10k
    )
    row = _queue_comparison(
        step_arr,
        y_arr,
        amount_arr,
        batch_selected,
        online_selected,
        alerts_per_10k=alerts_per_10k,
        contract="current_step_only",
    )
    return row, schedule


def batch_vs_backlog_capacity(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    *,
    alerts_per_10k: float,
) -> tuple[dict, pd.DataFrame]:
    """Compare whole-window hindsight top-k with a causal seen-so-far backlog."""
    step_arr, y_arr, score_arr, amount_arr, key_arr = _validate_inputs(
        step, y, score, amount, event_key
    )
    batch_selected = exact_capacity_mask(score_arr, key_arr, alerts_per_10k)
    online_selected, review_step, schedule = online_backlog_capacity_mask(
        step_arr, score_arr, key_arr, alerts_per_10k=alerts_per_10k
    )
    row = _queue_comparison(
        step_arr,
        y_arr,
        amount_arr,
        batch_selected,
        online_selected,
        alerts_per_10k=alerts_per_10k,
        contract="seen_so_far_backlog",
        review_step=review_step,
    )
    return row, schedule
