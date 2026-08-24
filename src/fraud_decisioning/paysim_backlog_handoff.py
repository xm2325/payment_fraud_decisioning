from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


HANDOFF_MODES = ("retain_old_scores", "rescore_pending", "drop_pending")


def _validate_inputs(
    step: Sequence[int],
    event_key: Sequence[int],
    regime_by_step: Mapping[int, int],
    score_by_regime: Mapping[int, Sequence[float]],
):
    step_arr = np.asarray(step, dtype=int)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if len(step_arr) == 0 or len(step_arr) != len(key_arr):
        raise ValueError("step and event_key must have equal non-zero length")
    regimes = sorted(int(value) for value in score_by_regime)
    if not regimes:
        raise ValueError("score_by_regime must not be empty")
    scores: dict[int, np.ndarray] = {}
    for regime in regimes:
        arr = np.asarray(score_by_regime[regime], dtype=float)
        if len(arr) != len(step_arr):
            raise ValueError("every regime score array must match step length")
        if np.any(~np.isfinite(arr)):
            raise ValueError("regime scores must be finite")
        scores[regime] = arr
    for value in np.unique(step_arr):
        if int(value) not in regime_by_step:
            raise ValueError(f"missing regime for step {int(value)}")
        if int(regime_by_step[int(value)]) not in scores:
            raise ValueError(f"unknown regime for step {int(value)}")
    return step_arr, key_arr, scores


def _final_entitlement(n: int, alerts_per_10k: float) -> int:
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    return min(int(n), int(np.floor(float(alerts_per_10k) * int(n) / 10_000)))


def continuous_backlog_handoff(
    step: Sequence[int],
    event_key: Sequence[int],
    regime_by_step: Mapping[int, int],
    score_by_regime: Mapping[int, Sequence[float]],
    *,
    alerts_per_10k: float,
    handoff_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Route one continuous backlog across model-refresh boundaries.

    Capacity entitlement is continuous across the full horizon. New transactions
    are scored only by the regime active when they arrive. At a regime change,
    pending cases either keep their current score, are rescored by the new
    regime, or are dropped. Selected cases are never reconsidered.

    The function is causal with respect to transaction arrival and model-release
    timing: a score from regime r cannot be used before the first step assigned
    to r. It does not assert that the upstream labels used to fit each regime
    were available under a real investigation-maturity process.
    """
    if handoff_mode not in HANDOFF_MODES:
        raise ValueError(f"handoff_mode must be one of {HANDOFF_MODES}")
    step_arr, key_arr, scores = _validate_inputs(
        step, event_key, regime_by_step, score_by_regime
    )

    selected = np.zeros(len(step_arr), dtype=bool)
    review_step = np.full(len(step_arr), -1, dtype=int)
    review_regime = np.full(len(step_arr), -1, dtype=int)
    pending_active = np.zeros(len(step_arr), dtype=bool)
    pending_score = np.full(len(step_arr), np.nan, dtype=float)
    pending_regime = np.full(len(step_arr), -1, dtype=int)

    heap: list[tuple[float, int, int]] = []
    cumulative_n = 0
    selected_total = 0
    previous_regime: int | None = None
    schedule_rows: list[dict] = []
    refresh_rows: list[dict] = []

    def rebuild_heap() -> None:
        nonlocal heap
        idx = np.flatnonzero(pending_active)
        heap = [
            (-float(pending_score[pos]), int(key_arr[pos]), int(pos)) for pos in idx
        ]
        heapq.heapify(heap)

    def pop_live() -> int | None:
        while heap:
            neg_score, key, pos = heapq.heappop(heap)
            if not pending_active[pos]:
                continue
            if int(key_arr[pos]) != key:
                continue
            if not np.isclose(-neg_score, pending_score[pos], rtol=0.0, atol=0.0):
                continue
            return int(pos)
        return None

    for value in np.unique(step_arr):
        value = int(value)
        current_regime = int(regime_by_step[value])
        is_refresh = previous_regime is not None and current_regime != previous_regime

        if is_refresh:
            pending_before = int(pending_active.sum())
            rescored = 0
            dropped = 0
            if handoff_mode == "rescore_pending":
                idx = np.flatnonzero(pending_active)
                if len(idx):
                    pending_score[idx] = scores[current_regime][idx]
                    pending_regime[idx] = current_regime
                    rescored = int(len(idx))
                rebuild_heap()
            elif handoff_mode == "drop_pending":
                dropped = pending_before
                pending_active[:] = False
                heap = []
            refresh_rows.append(
                {
                    "refresh_step": value,
                    "from_regime": int(previous_regime),
                    "to_regime": current_regime,
                    "handoff_mode": handoff_mode,
                    "pending_before": pending_before,
                    "rescored_pending": rescored,
                    "dropped_pending": dropped,
                    "pending_after_handoff": int(pending_active.sum()),
                    "selected_before_refresh": int(selected_total),
                }
            )

        idx = np.flatnonzero(step_arr == value)
        for pos in idx:
            score = float(scores[current_regime][pos])
            pending_active[pos] = True
            pending_score[pos] = score
            pending_regime[pos] = current_regime
            heapq.heappush(heap, (-score, int(key_arr[pos]), int(pos)))

        cumulative_n += len(idx)
        entitlement = _final_entitlement(cumulative_n, alerts_per_10k)
        available = max(0, entitlement - selected_total)
        selected_this_step = 0
        for _ in range(available):
            pos = pop_live()
            if pos is None:
                break
            pending_active[pos] = False
            selected[pos] = True
            review_step[pos] = value
            review_regime[pos] = current_regime
            selected_total += 1
            selected_this_step += 1

        schedule_rows.append(
            {
                "step": value,
                "active_regime": current_regime,
                "handoff_mode": handoff_mode,
                "transactions": int(len(idx)),
                "cumulative_transactions": int(cumulative_n),
                "cumulative_entitlement": int(entitlement),
                "new_slots_available": int(available),
                "selected_this_step": int(selected_this_step),
                "selected_cumulative": int(selected_total),
                "pending_after_selection": int(pending_active.sum()),
            }
        )
        previous_regime = current_regime

    final_entitlement = _final_entitlement(len(step_arr), alerts_per_10k)
    if selected_total != final_entitlement:
        raise AssertionError(
            f"Handoff strategy consumed {selected_total} reviews; expected {final_entitlement}"
        )
    if np.any(review_step[selected] < step_arr[selected]):
        raise AssertionError("review cannot precede transaction arrival")
    if np.any(review_regime[selected] < 0):
        raise AssertionError("every selected case must have a review regime")

    schedule = pd.DataFrame(schedule_rows)
    refresh = pd.DataFrame(refresh_rows)
    return selected, review_step, review_regime, schedule, refresh


def queue_metrics(
    step: Sequence[int],
    y: Sequence[int],
    amount: Sequence[float],
    selected: Sequence[bool],
    review_step: Sequence[int],
) -> dict[str, float | int]:
    step_arr = np.asarray(step, dtype=int)
    y_arr = np.asarray(y, dtype=int)
    amount_arr = np.asarray(amount, dtype=float)
    selected_arr = np.asarray(selected, dtype=bool)
    review_arr = np.asarray(review_step, dtype=int)
    n = len(step_arr)
    if any(len(value) != n for value in (y_arr, amount_arr, selected_arr, review_arr)):
        raise ValueError("queue metric inputs must have equal length")
    fraud = y_arr == 1
    alerts = int(selected_arr.sum())
    fraud_n = int(fraud.sum())
    fraud_alerts = int((selected_arr & fraud).sum())
    fraud_value_total = float(amount_arr[fraud].sum())
    fraud_value_captured = float(amount_arr[selected_arr & fraud].sum())
    delay = review_arr[selected_arr] - step_arr[selected_arr]
    if alerts and np.any(delay < 0):
        raise AssertionError("negative review delay")
    fraud_delay = delay[y_arr[selected_arr] == 1] if alerts else np.array([], dtype=int)
    recall = float(fraud_alerts / fraud_n) if fraud_n else math.nan
    value_recall = (
        float(fraud_value_captured / fraud_value_total) if fraud_value_total > 0 else math.nan
    )
    return {
        "alerts": alerts,
        "fraud_alerts": fraud_alerts,
        "precision": float(fraud_alerts / alerts) if alerts else math.nan,
        "fraud_recall": recall,
        "fraud_value_recall": value_recall,
        "balanced_hmean": (
            float(2 * recall * value_recall / (recall + value_recall))
            if recall + value_recall > 0
            else 0.0
        ),
        "fraud_value_captured": fraud_value_captured,
        "review_delay_mean_steps": float(np.mean(delay)) if alerts else math.nan,
        "review_delay_p50_steps": float(np.quantile(delay, 0.50)) if alerts else math.nan,
        "review_delay_p90_steps": float(np.quantile(delay, 0.90)) if alerts else math.nan,
        "review_delay_max_steps": int(np.max(delay)) if alerts else 0,
        "fraud_review_delay_mean_steps": (
            float(np.mean(fraud_delay)) if len(fraud_delay) else math.nan
        ),
        "fraud_review_delay_p90_steps": (
            float(np.quantile(fraud_delay, 0.90)) if len(fraud_delay) else math.nan
        ),
    }


def queue_overlap(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int]:
    left = np.asarray(a, dtype=bool)
    right = np.asarray(b, dtype=bool)
    if len(left) == 0 or len(left) != len(right):
        raise ValueError("queue masks must have equal non-zero length")
    overlap = int((left & right).sum())
    union = int((left | right).sum())
    left_n = int(left.sum())
    right_n = int(right.sum())
    return {
        "left_alerts": left_n,
        "right_alerts": right_n,
        "overlap": overlap,
        "overlap_rate_left": float(overlap / left_n) if left_n else math.nan,
        "overlap_rate_right": float(overlap / right_n) if right_n else math.nan,
        "jaccard": float(overlap / union) if union else 1.0,
        "replacement_count": int((right & ~left).sum()),
    }
