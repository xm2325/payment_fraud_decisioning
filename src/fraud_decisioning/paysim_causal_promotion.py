from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from .paysim_online_capacity import online_backlog_capacity_mask


METRICS = ("precision", "fraud_recall", "fraud_value_recall", "balanced_hmean")


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
    if n == 0 or any(len(value) != n for value in arrays):
        raise ValueError("All causal promotion inputs must have equal non-zero length")
    if np.any((arrays[1] != 0) & (arrays[1] != 1)):
        raise ValueError("y must contain only 0/1 labels")
    if np.any(~np.isfinite(arrays[2])) or np.any(arrays[2] < 0):
        raise ValueError("amount must be finite and non-negative")
    if np.any(~np.isfinite(arrays[4])) or np.any(~np.isfinite(arrays[5])):
        raise ValueError("policy scores must be finite")
    return arrays


def _metrics(y: np.ndarray, amount: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    fraud = y == 1
    alerts = int(selected.sum())
    fraud_n = int(fraud.sum())
    fraud_alerts = int((selected & fraud).sum())
    fraud_value_total = float(amount[fraud].sum())
    fraud_value_captured = float(amount[selected & fraud].sum())
    precision = float(fraud_alerts / alerts) if alerts else math.nan
    recall = float(fraud_alerts / fraud_n) if fraud_n else math.nan
    value_recall = (
        float(fraud_value_captured / fraud_value_total)
        if fraud_value_total > 0
        else math.nan
    )
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
        "fraud_value_captured": fraud_value_captured,
    }


def _delay_summary(
    step: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    review_step: np.ndarray,
) -> dict[str, float | int]:
    if not selected.any():
        return {
            "review_delay_mean_steps": math.nan,
            "review_delay_p90_steps": math.nan,
            "review_delay_max_steps": 0,
            "fraud_review_delay_mean_steps": math.nan,
            "fraud_review_delay_p90_steps": math.nan,
        }
    delay = review_step[selected] - step[selected]
    if np.any(delay < 0):
        raise AssertionError("review step cannot precede arrival step")
    fraud_delay = delay[y[selected] == 1]
    return {
        "review_delay_mean_steps": float(np.mean(delay)),
        "review_delay_p90_steps": float(np.quantile(delay, 0.90)),
        "review_delay_max_steps": int(np.max(delay)),
        "fraud_review_delay_mean_steps": (
            float(np.mean(fraud_delay)) if len(fraud_delay) else math.nan
        ),
        "fraud_review_delay_p90_steps": (
            float(np.quantile(fraud_delay, 0.90)) if len(fraud_delay) else math.nan
        ),
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
        incumbent = mask & incumbent_selected
        candidate = mask & candidate_selected
        rows[i] = (
            float(fraud.sum()),
            float(amount[fraud].sum()),
            float(incumbent.sum()),
            float((incumbent & fraud_mask).sum()),
            float(amount[incumbent & fraud_mask].sum()),
            float(candidate.sum()),
            float((candidate & fraud_mask).sum()),
            float(amount[candidate & fraud_mask].sum()),
        )
    return rows


def _metrics_from_aggregate(total: np.ndarray) -> tuple[dict[str, float], dict[str, float]] | None:
    fraud_n, fraud_value, inc_alerts, inc_fraud, inc_value, cand_alerts, cand_fraud, cand_value = total
    if fraud_n <= 0 or fraud_value <= 0 or inc_alerts <= 0 or cand_alerts <= 0:
        return None

    def one(alerts: float, caught_n: float, caught_value: float) -> dict[str, float]:
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


def paired_causal_backlog_block_bootstrap(
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
    """Paired block uncertainty for realised causal backlog queues.

    The point queues are constructed strictly forward with the seen-so-far
    backlog contract. Bootstrap resampling is then applied to paired per-step
    realised outcomes, matching the scope of the earlier v1.9 uncertainty gate:
    it quantifies temporal uncertainty conditional on the observed frozen queues.
    It does not rerun model fitting or reconstruct a different queue for each
    resampled history.
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

    incumbent_selected, incumbent_review_step, _ = online_backlog_capacity_mask(
        step_arr, inc_score, key_arr, alerts_per_10k=alerts_per_10k
    )
    candidate_selected, candidate_review_step, _ = online_backlog_capacity_mask(
        step_arr, cand_score, key_arr, alerts_per_10k=alerts_per_10k
    )
    if incumbent_selected.sum() != candidate_selected.sum():
        raise AssertionError("Incumbent and candidate must consume identical causal capacity")

    incumbent = _metrics(y_arr, amount_arr, incumbent_selected)
    candidate = _metrics(y_arr, amount_arr, candidate_selected)
    incumbent.update(_delay_summary(step_arr, y_arr, incumbent_selected, incumbent_review_step))
    candidate.update(_delay_summary(step_arr, y_arr, candidate_selected, candidate_review_step))
    point_delta = {
        metric: float(candidate[metric] - incumbent[metric]) for metric in METRICS
    }

    per_step = _step_aggregates(
        step_arr, y_arr, amount_arr, incumbent_selected, candidate_selected
    )
    n_steps = len(per_step)
    n_blocks = int(np.ceil(n_steps / block_steps))
    offsets = np.arange(block_steps, dtype=int)
    rng = np.random.default_rng(seed)
    boot = {metric: [] for metric in METRICS}

    for _ in range(n_bootstrap):
        starts = rng.integers(0, n_steps, size=n_blocks)
        positions = ((starts[:, None] + offsets[None, :]) % n_steps).reshape(-1)[:n_steps]
        pair = _metrics_from_aggregate(per_step[positions].sum(axis=0))
        if pair is None:
            continue
        inc_rep, cand_rep = pair
        for metric in METRICS:
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

    overlap = int((incumbent_selected & candidate_selected).sum())
    alerts = int(incumbent_selected.sum())
    union = int((incumbent_selected | candidate_selected).sum())
    return {
        "routing_contract": "seen_so_far_backlog",
        "incumbent": incumbent,
        "candidate": candidate,
        "delta_intervals": intervals,
        "queue_overlap": overlap,
        "queue_overlap_rate": float(overlap / alerts) if alerts else math.nan,
        "queue_jaccard": float(overlap / union) if union else 1.0,
        "replacement_count": int((candidate_selected & ~incumbent_selected).sum()),
        "n_test_steps": int(n_steps),
        "block_steps": int(block_steps),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_valid": int(valid),
        "tail_alpha": float(tail_alpha),
        "seed": int(seed),
        "uncertainty_scope": "paired realised causal-queue outcomes conditional on observed frozen queues",
    }
