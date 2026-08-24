from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import numpy as np


PROFILE_PRIMARY_METRIC = {
    "case_first": "fraud_recall",
    "balanced": "balanced_hmean",
    "value_first": "fraud_value_recall",
}


@dataclass(frozen=True)
class PromotionGateConfig:
    alerts_per_10k: float = 50.0
    block_steps: int = 5
    n_bootstrap: int = 2000
    family_alpha: float = 0.05
    min_primary_gain: float = 0.02
    precision_noninferiority_margin: float = 0.02
    recall_noninferiority_margin: float = 0.02
    value_recall_noninferiority_margin: float = 0.02
    seed: int = 20260824


def _as_arrays(step, y, amount, event_key, incumbent_score, candidate_score):
    arrays = (
        np.asarray(step, dtype=int),
        np.asarray(y, dtype=int),
        np.asarray(amount, dtype=float),
        np.asarray(event_key, dtype=np.uint64),
        np.asarray(incumbent_score, dtype=float),
        np.asarray(candidate_score, dtype=float),
    )
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        raise ValueError("All policy-comparison inputs must have equal non-zero length")
    if np.any(~np.isfinite(arrays[2])) or np.any(arrays[2] < 0):
        raise ValueError("amount must be finite and non-negative")
    if np.any(~np.isfinite(arrays[4])) or np.any(~np.isfinite(arrays[5])):
        raise ValueError("policy scores must be finite")
    return arrays


def exact_capacity_mask(score: Sequence[float], event_key: Sequence[int], alerts_per_10k: float) -> np.ndarray:
    """Return the deterministic exact top-k queue used by the routing contract."""
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    score_arr = np.asarray(score, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if len(score_arr) == 0 or len(score_arr) != len(key_arr):
        raise ValueError("score and event_key must have equal non-zero length")
    if np.any(~np.isfinite(score_arr)):
        raise ValueError("score must be finite")
    k = min(len(score_arr), int(np.floor(float(alerts_per_10k) * len(score_arr) / 10_000)))
    selected = np.zeros(len(score_arr), dtype=bool)
    if k:
        order = np.lexsort((key_arr, -score_arr))
        selected[order[:k]] = True
    return selected


def _metrics(y: np.ndarray, amount: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    fraud = y == 1
    alerts = int(selected.sum())
    fraud_n = int(fraud.sum())
    fraud_value = float(amount[fraud].sum())
    fraud_alerts = int((selected & fraud).sum())
    captured_value = float(amount[selected & fraud].sum())
    precision = float(fraud_alerts / alerts) if alerts else math.nan
    recall = float(fraud_alerts / fraud_n) if fraud_n else math.nan
    value_recall = float(captured_value / fraud_value) if fraud_value > 0 else math.nan
    balanced_hmean = (
        float(2 * recall * value_recall / (recall + value_recall))
        if recall + value_recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "fraud_recall": recall,
        "fraud_value_recall": value_recall,
        "balanced_hmean": balanced_hmean,
        "alerts": alerts,
        "fraud_alerts": fraud_alerts,
    }


def _step_aggregates(
    step: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    incumbent_selected: np.ndarray,
    candidate_selected: np.ndarray,
) -> np.ndarray:
    unique_steps = np.unique(step)
    rows = np.zeros((len(unique_steps), 8), dtype=float)
    fraud_mask = y == 1
    for i, value in enumerate(unique_steps):
        mask = step == value
        fraud = mask & fraud_mask
        inc = mask & incumbent_selected
        cand = mask & candidate_selected
        rows[i] = (
            float(fraud.sum()),
            float(amount[fraud].sum()),
            float(inc.sum()),
            float((inc & fraud_mask).sum()),
            float(amount[inc & fraud_mask].sum()),
            float(cand.sum()),
            float((cand & fraud_mask).sum()),
            float(amount[cand & fraud_mask].sum()),
        )
    return rows


def _metrics_from_aggregate(total: np.ndarray) -> tuple[dict[str, float], dict[str, float]] | None:
    fraud_n, fraud_value, inc_alerts, inc_fraud, inc_value, cand_alerts, cand_fraud, cand_value = total
    if fraud_n <= 0 or fraud_value <= 0 or inc_alerts <= 0 or cand_alerts <= 0:
        return None

    def one(alerts, caught_n, caught_value):
        recall = float(caught_n / fraud_n)
        value_recall = float(caught_value / fraud_value)
        return {
            "precision": float(caught_n / alerts),
            "fraud_recall": recall,
            "fraud_value_recall": value_recall,
            "balanced_hmean": (
                float(2 * recall * value_recall / (recall + value_recall))
                if recall + value_recall > 0
                else 0.0
            ),
        }

    return one(inc_alerts, inc_fraud, inc_value), one(cand_alerts, cand_fraud, cand_value)


def paired_circular_block_bootstrap(
    step: Sequence[int],
    y: Sequence[int],
    amount: Sequence[float],
    event_key: Sequence[int],
    incumbent_score: Sequence[float],
    candidate_score: Sequence[float],
    *,
    alerts_per_10k: float = 50.0,
    block_steps: int = 5,
    n_bootstrap: int = 2000,
    tail_alpha: float = 0.025,
    seed: int = 20260824,
) -> dict:
    """Paired time-block bootstrap for two frozen exact-capacity policies.

    Policy selections are fixed on the observed test window. The bootstrap then
    resamples contiguous time blocks, preserving temporal dependence while
    comparing the same realised outcomes for incumbent and candidate queues.
    It measures uncertainty in incremental captured outcomes, not model-fit
    uncertainty and not future production performance.
    """
    if block_steps < 1 or n_bootstrap < 100:
        raise ValueError("block_steps must be positive and n_bootstrap must be at least 100")
    if not 0 < tail_alpha < 0.5:
        raise ValueError("tail_alpha must be between 0 and 0.5")

    step_arr, y_arr, amount_arr, key_arr, inc_score, cand_score = _as_arrays(
        step, y, amount, event_key, incumbent_score, candidate_score
    )
    unique_steps = np.unique(step_arr)
    if len(unique_steps) < block_steps:
        raise ValueError("Need at least block_steps distinct test steps")

    inc_selected = exact_capacity_mask(inc_score, key_arr, alerts_per_10k)
    cand_selected = exact_capacity_mask(cand_score, key_arr, alerts_per_10k)
    incumbent = _metrics(y_arr, amount_arr, inc_selected)
    candidate = _metrics(y_arr, amount_arr, cand_selected)
    point_delta = {
        metric: float(candidate[metric] - incumbent[metric])
        for metric in ("precision", "fraud_recall", "fraud_value_recall", "balanced_hmean")
    }

    per_step = _step_aggregates(step_arr, y_arr, amount_arr, inc_selected, cand_selected)
    n_steps = len(per_step)
    n_blocks = int(np.ceil(n_steps / block_steps))
    offsets = np.arange(block_steps, dtype=int)
    rng = np.random.default_rng(seed)
    boot = {metric: [] for metric in point_delta}

    for _ in range(n_bootstrap):
        starts = rng.integers(0, n_steps, size=n_blocks)
        positions = ((starts[:, None] + offsets[None, :]) % n_steps).reshape(-1)[:n_steps]
        pair = _metrics_from_aggregate(per_step[positions].sum(axis=0))
        if pair is None:
            continue
        inc_rep, cand_rep = pair
        for metric in boot:
            boot[metric].append(float(cand_rep[metric] - inc_rep[metric]))

    valid = min(len(values) for values in boot.values())
    if valid < max(100, int(0.9 * n_bootstrap)):
        raise ValueError("Too few valid bootstrap replicates")

    intervals = {}
    for metric, values in boot.items():
        arr = np.asarray(values, dtype=float)
        intervals[metric] = {
            "point_delta": point_delta[metric],
            "lower": float(np.quantile(arr, tail_alpha)),
            "upper": float(np.quantile(arr, 1 - tail_alpha)),
            "bootstrap_median": float(np.median(arr)),
        }

    return {
        "incumbent": incumbent,
        "candidate": candidate,
        "delta_intervals": intervals,
        "n_test_steps": int(n_steps),
        "block_steps": int(block_steps),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(valid),
        "tail_alpha": float(tail_alpha),
        "seed": int(seed),
    }


def promotion_decision(
    profile: str,
    incumbent_alpha: float,
    candidate_alpha: float,
    uncertainty: dict | None,
    *,
    policy_changed: bool = True,
    min_primary_gain: float = 0.02,
    precision_noninferiority_margin: float = 0.02,
    recall_noninferiority_margin: float = 0.02,
    value_recall_noninferiority_margin: float = 0.02,
) -> dict:
    """Apply a pre-declared promotion rule to family-adjusted lower bounds."""
    if profile not in PROFILE_PRIMARY_METRIC:
        raise ValueError(f"Unknown profile: {profile}")
    if not policy_changed:
        return {
            "decision": "NO_POLICY_CHANGE",
            "primary_metric": PROFILE_PRIMARY_METRIC[profile],
            "reason": "candidate and incumbent policy scores are identical by construction",
        }
    if uncertainty is None:
        raise ValueError("uncertainty is required for a changed candidate policy")

    primary = PROFILE_PRIMARY_METRIC[profile]
    intervals = uncertainty["delta_intervals"]
    primary_point = float(intervals[primary]["point_delta"])
    primary_lower = float(intervals[primary]["lower"])

    guards = {
        "precision": float(intervals["precision"]["lower"]) >= -precision_noninferiority_margin,
    }
    if primary != "fraud_recall":
        guards["fraud_recall"] = (
            float(intervals["fraud_recall"]["lower"]) >= -recall_noninferiority_margin
        )
    if primary != "fraud_value_recall":
        guards["fraud_value_recall"] = (
            float(intervals["fraud_value_recall"]["lower"])
            >= -value_recall_noninferiority_margin
        )

    promote = primary_point >= min_primary_gain and primary_lower > 0 and all(guards.values())
    reason = (
        "family-adjusted primary lower bound is positive and all guardrails pass"
        if promote
        else "promotion evidence is insufficient under the pre-declared gain and guardrail rules"
    )
    return {
        "decision": "PROMOTE" if promote else "KEEP_INCUMBENT",
        "primary_metric": primary,
        "primary_point_delta": primary_point,
        "primary_lower_bound": primary_lower,
        "minimum_primary_gain": float(min_primary_gain),
        "incumbent_alpha": float(incumbent_alpha),
        "candidate_alpha": float(candidate_alpha),
        "guardrails": guards,
        "reason": reason,
    }
