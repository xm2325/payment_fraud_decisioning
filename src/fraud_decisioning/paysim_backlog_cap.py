from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


CAP_GRID: tuple[int | None, ...] = (0, 5_000, 10_000, 25_000, 50_000, 100_000, None)


def cap_label(max_pending_cases: int | None) -> str:
    if max_pending_cases is None:
        return "infinite"
    if int(max_pending_cases) < 0:
        raise ValueError("max_pending_cases must be non-negative or None")
    return str(int(max_pending_cases))


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
        value = int(value)
        if value not in regime_by_step:
            raise ValueError(f"missing regime for step {value}")
        if int(regime_by_step[value]) not in scores:
            raise ValueError(f"unknown regime for step {value}")
    return step_arr, key_arr, scores


def _entitlement(n: int, alerts_per_10k: float) -> int:
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    return min(int(n), int(np.floor(float(alerts_per_10k) * int(n) / 10_000)))


def continuous_rescore_bounded_backlog(
    step: Sequence[int],
    event_key: Sequence[int],
    regime_by_step: Mapping[int, int],
    score_by_regime: Mapping[int, Sequence[float]],
    *,
    alerts_per_10k: float,
    max_pending_cases: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run a causal rescore-on-refresh backlog with a hard pending-case cap.

    New arrivals and previously pending cases compete for newly earned analyst
    capacity before the memory cap is applied. If the remaining pending pool is
    above the cap, the lowest current-score cases are evicted; equal-score
    eviction removes larger event keys first so the surviving pool matches the
    deterministic top-priority ordering used for review selection.

    At a model refresh, every surviving pending case is rescored by the newly
    released regime before later arrivals are processed. A later-regime score is
    never used before its release step. Fraud labels are not routing inputs.
    """
    if max_pending_cases is not None and int(max_pending_cases) < 0:
        raise ValueError("max_pending_cases must be non-negative or None")
    cap = None if max_pending_cases is None else int(max_pending_cases)
    step_arr, key_arr, scores = _validate_inputs(
        step, event_key, regime_by_step, score_by_regime
    )

    selected = np.zeros(len(step_arr), dtype=bool)
    evicted = np.zeros(len(step_arr), dtype=bool)
    review_step = np.full(len(step_arr), -1, dtype=int)
    review_regime = np.full(len(step_arr), -1, dtype=int)
    pending_active = np.zeros(len(step_arr), dtype=bool)
    pending_score = np.full(len(step_arr), np.nan, dtype=float)

    review_heap: list[tuple[float, int, int]] = []
    eviction_heap: list[tuple[float, int, int]] = []
    pending_count = 0
    cumulative_n = 0
    selected_total = 0
    evicted_total = 0
    previous_regime: int | None = None
    schedule_rows: list[dict] = []
    refresh_rows: list[dict] = []

    def rebuild_heaps() -> None:
        nonlocal review_heap, eviction_heap
        idx = np.flatnonzero(pending_active)
        review_heap = [
            (-float(pending_score[pos]), int(key_arr[pos]), int(pos)) for pos in idx
        ]
        eviction_heap = [
            (float(pending_score[pos]), -int(key_arr[pos]), int(pos)) for pos in idx
        ]
        heapq.heapify(review_heap)
        heapq.heapify(eviction_heap)

    def pop_review() -> int | None:
        while review_heap:
            neg_score, key, pos = heapq.heappop(review_heap)
            if not pending_active[pos]:
                continue
            if int(key_arr[pos]) != key:
                continue
            if not np.isclose(-neg_score, pending_score[pos], rtol=0.0, atol=0.0):
                continue
            return int(pos)
        return None

    def pop_eviction() -> int | None:
        while eviction_heap:
            score, neg_key, pos = heapq.heappop(eviction_heap)
            if not pending_active[pos]:
                continue
            if -int(key_arr[pos]) != neg_key:
                continue
            if not np.isclose(score, pending_score[pos], rtol=0.0, atol=0.0):
                continue
            return int(pos)
        return None

    for value in np.unique(step_arr):
        value = int(value)
        current_regime = int(regime_by_step[value])
        is_refresh = previous_regime is not None and current_regime != previous_regime

        if is_refresh:
            idx = np.flatnonzero(pending_active)
            if len(idx):
                pending_score[idx] = scores[current_regime][idx]
            rebuild_heaps()
            refresh_rows.append(
                {
                    "refresh_step": value,
                    "from_regime": int(previous_regime),
                    "to_regime": current_regime,
                    "max_pending_cases": cap_label(cap),
                    "pending_rescored": int(len(idx)),
                    "selected_before_refresh": int(selected_total),
                }
            )

        arrivals = np.flatnonzero(step_arr == value)
        for pos in arrivals:
            score = float(scores[current_regime][pos])
            pending_active[pos] = True
            pending_score[pos] = score
            heapq.heappush(review_heap, (-score, int(key_arr[pos]), int(pos)))
            heapq.heappush(eviction_heap, (score, -int(key_arr[pos]), int(pos)))
            pending_count += 1

        cumulative_n += len(arrivals)
        entitlement = _entitlement(cumulative_n, alerts_per_10k)
        available = max(0, entitlement - selected_total)
        selected_this_step = 0
        for _ in range(available):
            pos = pop_review()
            if pos is None:
                break
            pending_active[pos] = False
            selected[pos] = True
            review_step[pos] = value
            review_regime[pos] = current_regime
            pending_count -= 1
            selected_total += 1
            selected_this_step += 1

        evicted_this_step = 0
        if cap is not None:
            while pending_count > cap:
                pos = pop_eviction()
                if pos is None:
                    raise AssertionError("bounded backlog could not locate a live eviction candidate")
                pending_active[pos] = False
                evicted[pos] = True
                pending_count -= 1
                evicted_total += 1
                evicted_this_step += 1

        if pending_count != int(pending_active.sum()):
            raise AssertionError("pending-count bookkeeping drifted")
        schedule_rows.append(
            {
                "step": value,
                "active_regime": current_regime,
                "max_pending_cases": cap_label(cap),
                "transactions": int(len(arrivals)),
                "cumulative_transactions": int(cumulative_n),
                "cumulative_entitlement": int(entitlement),
                "new_slots_available": int(available),
                "selected_this_step": int(selected_this_step),
                "selected_cumulative": int(selected_total),
                "evicted_this_step": int(evicted_this_step),
                "evicted_cumulative": int(evicted_total),
                "pending_after_selection_and_cap": int(pending_count),
            }
        )
        previous_regime = current_regime

    final_entitlement = _entitlement(len(step_arr), alerts_per_10k)
    if selected_total != final_entitlement:
        raise AssertionError(
            f"bounded strategy consumed {selected_total} reviews; expected {final_entitlement}"
        )
    if np.any(review_step[selected] < step_arr[selected]):
        raise AssertionError("review cannot precede transaction arrival")
    if np.any(selected & evicted):
        raise AssertionError("a case cannot be both selected and evicted")
    if cap is not None and int(pending_active.sum()) > cap:
        raise AssertionError("final pending pool exceeds the declared cap")

    return (
        selected,
        review_step,
        review_regime,
        evicted,
        pd.DataFrame(schedule_rows),
        pd.DataFrame(refresh_rows),
    )


def eviction_metrics(
    y: Sequence[int],
    amount: Sequence[float],
    evicted: Sequence[bool],
) -> dict[str, float | int]:
    """Retrospective label diagnostics for score-based memory evictions."""
    y_arr = np.asarray(y, dtype=int)
    amount_arr = np.asarray(amount, dtype=float)
    evicted_arr = np.asarray(evicted, dtype=bool)
    if len(y_arr) == 0 or len(y_arr) != len(amount_arr) or len(y_arr) != len(evicted_arr):
        raise ValueError("eviction metric inputs must have equal non-zero length")
    fraud = y_arr == 1
    fraud_n = int(fraud.sum())
    fraud_value_total = float(amount_arr[fraud].sum())
    evicted_fraud = int((evicted_arr & fraud).sum())
    evicted_fraud_value = float(amount_arr[evicted_arr & fraud].sum())
    return {
        "evicted_cases": int(evicted_arr.sum()),
        "evicted_rate": float(evicted_arr.mean()),
        "evicted_fraud_cases": evicted_fraud,
        "evicted_fraud_case_share": float(evicted_fraud / fraud_n) if fraud_n else math.nan,
        "evicted_fraud_value": evicted_fraud_value,
        "evicted_fraud_value_share": (
            float(evicted_fraud_value / fraud_value_total) if fraud_value_total > 0 else math.nan
        ),
    }
