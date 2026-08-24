from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TTL_GRID: tuple[int | None, ...] = (0, 5, 10, 20, 40, None)


def ttl_label(max_age_steps: int | None) -> str:
    if max_age_steps is None:
        return "infinite"
    if int(max_age_steps) < 0:
        raise ValueError("max_age_steps must be non-negative or None")
    return str(int(max_age_steps))


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


def continuous_rescore_backlog_ttl(
    step: Sequence[int],
    event_key: Sequence[int],
    regime_by_step: Mapping[int, int],
    score_by_regime: Mapping[int, Sequence[float]],
    *,
    alerts_per_10k: float,
    max_age_steps: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run a continuous causal rescore-on-refresh backlog with an age TTL.

    A pending case is eligible at step ``t`` iff ``t - arrival_step <= TTL``.
    Expiry happens before a model refresh at that step, so the TTL can reduce
    refresh-time rescoring workload. ``None`` means no age expiry. New cases are
    scored only by the model regime active on arrival; surviving pending cases
    are rescored only when a new regime is released. No later-regime score is
    used before its release step.

    Labels are not inputs to routing or expiry. Returned expiry masks are for
    retrospective diagnostics only.
    """
    if max_age_steps is not None and int(max_age_steps) < 0:
        raise ValueError("max_age_steps must be non-negative or None")
    ttl = None if max_age_steps is None else int(max_age_steps)
    step_arr, key_arr, scores = _validate_inputs(
        step, event_key, regime_by_step, score_by_regime
    )

    selected = np.zeros(len(step_arr), dtype=bool)
    expired = np.zeros(len(step_arr), dtype=bool)
    review_step = np.full(len(step_arr), -1, dtype=int)
    review_regime = np.full(len(step_arr), -1, dtype=int)
    pending_active = np.zeros(len(step_arr), dtype=bool)
    pending_score = np.full(len(step_arr), np.nan, dtype=float)

    heap: list[tuple[float, int, int]] = []
    cumulative_n = 0
    selected_total = 0
    expired_total = 0
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

        pending_before_expiry = int(pending_active.sum())
        expired_this_step = 0
        if ttl is not None and pending_before_expiry:
            pending_idx = np.flatnonzero(pending_active)
            too_old = pending_idx[(value - step_arr[pending_idx]) > ttl]
            if len(too_old):
                pending_active[too_old] = False
                expired[too_old] = True
                expired_this_step = int(len(too_old))
                expired_total += expired_this_step

        pending_after_expiry = int(pending_active.sum())
        if is_refresh:
            idx = np.flatnonzero(pending_active)
            if len(idx):
                pending_score[idx] = scores[current_regime][idx]
            rebuild_heap()
            refresh_rows.append(
                {
                    "refresh_step": value,
                    "from_regime": int(previous_regime),
                    "to_regime": current_regime,
                    "ttl_steps": ttl_label(ttl),
                    "pending_before_expiry": pending_before_expiry,
                    "expired_before_refresh": expired_this_step,
                    "pending_rescored": int(len(idx)),
                    "rescore_reduction_vs_unexpired_pool": int(
                        pending_before_expiry - len(idx)
                    ),
                    "selected_before_refresh": int(selected_total),
                }
            )

        arrivals = np.flatnonzero(step_arr == value)
        for pos in arrivals:
            score = float(scores[current_regime][pos])
            pending_active[pos] = True
            pending_score[pos] = score
            heapq.heappush(heap, (-score, int(key_arr[pos]), int(pos)))

        cumulative_n += len(arrivals)
        entitlement = _entitlement(cumulative_n, alerts_per_10k)
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

        pending_idx = np.flatnonzero(pending_active)
        pending_age = value - step_arr[pending_idx] if len(pending_idx) else np.array([], dtype=int)
        schedule_rows.append(
            {
                "step": value,
                "active_regime": current_regime,
                "ttl_steps": ttl_label(ttl),
                "transactions": int(len(arrivals)),
                "cumulative_transactions": int(cumulative_n),
                "cumulative_entitlement": int(entitlement),
                "new_slots_available": int(available),
                "selected_this_step": int(selected_this_step),
                "selected_cumulative": int(selected_total),
                "expired_this_step": int(expired_this_step),
                "expired_cumulative": int(expired_total),
                "pending_after_selection": int(len(pending_idx)),
                "pending_age_mean_steps": float(np.mean(pending_age)) if len(pending_age) else 0.0,
                "pending_age_p90_steps": float(np.quantile(pending_age, 0.90)) if len(pending_age) else 0.0,
                "pending_age_max_steps": int(np.max(pending_age)) if len(pending_age) else 0,
            }
        )
        previous_regime = current_regime

    final_entitlement = _entitlement(len(step_arr), alerts_per_10k)
    if selected_total != final_entitlement:
        raise AssertionError(
            f"TTL strategy consumed {selected_total} reviews; expected {final_entitlement}"
        )
    if np.any(review_step[selected] < step_arr[selected]):
        raise AssertionError("review cannot precede transaction arrival")
    if ttl is not None and np.any((review_step[selected] - step_arr[selected]) > ttl):
        raise AssertionError("selected review exceeded the declared TTL")
    if np.any(selected & expired):
        raise AssertionError("a case cannot be both selected and expired")

    return (
        selected,
        review_step,
        review_regime,
        expired,
        pd.DataFrame(schedule_rows),
        pd.DataFrame(refresh_rows),
    )


def expiry_metrics(
    y: Sequence[int],
    amount: Sequence[float],
    expired: Sequence[bool],
) -> dict[str, float | int]:
    """Retrospective label-based diagnostics for cases expired by a label-free TTL."""
    y_arr = np.asarray(y, dtype=int)
    amount_arr = np.asarray(amount, dtype=float)
    expired_arr = np.asarray(expired, dtype=bool)
    if len(y_arr) == 0 or len(y_arr) != len(amount_arr) or len(y_arr) != len(expired_arr):
        raise ValueError("expiry metric inputs must have equal non-zero length")
    fraud = y_arr == 1
    fraud_n = int(fraud.sum())
    total_fraud_value = float(amount_arr[fraud].sum())
    expired_fraud = int((expired_arr & fraud).sum())
    expired_fraud_value = float(amount_arr[expired_arr & fraud].sum())
    return {
        "expired_cases": int(expired_arr.sum()),
        "expired_rate": float(expired_arr.mean()),
        "expired_fraud_cases": expired_fraud,
        "expired_fraud_case_share": float(expired_fraud / fraud_n) if fraud_n else math.nan,
        "expired_fraud_value": expired_fraud_value,
        "expired_fraud_value_share": (
            float(expired_fraud_value / total_fraud_value) if total_fraud_value > 0 else math.nan
        ),
    }
