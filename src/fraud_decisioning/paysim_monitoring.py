from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .paysim_metrics import rule_metrics, threshold_at_legit_rate


RECIPIENT_SIGNAL_COLUMNS = [
    "recipient_fanin_24h",
    "recipient_tx_7d",
    "recipient_amount_24h",
    "recipient_unique_senders_7d",
]


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else math.nan


def _locked_threshold_row(
    period: str,
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    threshold: float,
    target_legit_flag_rate: float,
) -> dict:
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    pred = score_arr >= threshold
    fraud = y_arr == 1
    legit = ~fraud
    tp = int((pred & fraud).sum())
    fp = int((pred & legit).sum())
    alerts = int(pred.sum())
    legit_flag_rate = _safe_ratio(fp, int(legit.sum()))
    fraud_value_total = float(amount_arr[fraud].sum())
    fraud_value_captured = float(amount_arr[pred & fraud].sum())
    return {
        "period": period,
        "n": int(len(y_arr)),
        "fraud_n": int(fraud.sum()),
        "fraud_rate": float(y_arr.mean()),
        "threshold": float(threshold),
        "alerts": alerts,
        "alert_rate": _safe_ratio(alerts, len(y_arr)),
        "precision": float(y_arr[pred].mean()) if alerts else math.nan,
        "recall": _safe_ratio(tp, int(fraud.sum())),
        "legit_flag_rate": legit_flag_rate,
        "legit_alerts_per_10k": float(legit_flag_rate * 10_000),
        "budget_multiplier_vs_target": _safe_ratio(legit_flag_rate, target_legit_flag_rate),
        "fraud_value_recall": _safe_ratio(fraud_value_captured, fraud_value_total),
        "score_p50": float(np.quantile(score_arr, 0.50)),
        "score_p90": float(np.quantile(score_arr, 0.90)),
        "score_p99": float(np.quantile(score_arr, 0.99)),
        "score_p999": float(np.quantile(score_arr, 0.999)),
    }


def locked_threshold_drift(
    val_y: Sequence[int],
    val_score: Sequence[float],
    val_amount: Sequence[float],
    future_y: Sequence[int],
    future_score: Sequence[float],
    future_amount: Sequence[float],
    threshold: float,
    target_legit_flag_rate: float = 0.001,
) -> pd.DataFrame:
    """Measure realised behaviour after locking a validation-selected scalar threshold."""
    rows = [
        _locked_threshold_row(
            "validation", val_y, val_score, val_amount, threshold, target_legit_flag_rate
        ),
        _locked_threshold_row(
            "future_test", future_y, future_score, future_amount, threshold, target_legit_flag_rate
        ),
    ]
    val_lfr = rows[0]["legit_flag_rate"]
    rows[0]["budget_multiplier_vs_validation_actual"] = 1.0
    rows[1]["budget_multiplier_vs_validation_actual"] = _safe_ratio(
        rows[1]["legit_flag_rate"], val_lfr
    )
    return pd.DataFrame(rows)


def future_budget_windows(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    threshold: float,
    target_legit_flag_rate: float = 0.001,
    n_windows: int = 3,
) -> pd.DataFrame:
    """Evaluate a locked scalar threshold across contiguous future step windows."""
    step_arr = np.asarray(step, dtype=int)
    unique_steps = np.unique(step_arr)
    if len(unique_steps) < n_windows:
        raise ValueError("Need at least one distinct step per monitoring window")
    chunks = [c for c in np.array_split(unique_steps, n_windows) if len(c)]
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    rows: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        mask = np.isin(step_arr, chunk)
        row = _locked_threshold_row(
            f"future_window_{idx}",
            y_arr[mask],
            score_arr[mask],
            amount_arr[mask],
            threshold,
            target_legit_flag_rate,
        )
        row["step_min"] = int(chunk.min())
        row["step_max"] = int(chunk.max())
        rows.append(row)
    return pd.DataFrame(rows)


def posthoc_threshold_cap(
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    target_legit_flag_rate: float = 0.001,
) -> dict:
    """Retrospective scalar-threshold hard-cap diagnostic; it may under-use budget when scores tie."""
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    threshold = threshold_at_legit_rate(y_arr, score_arr, target_legit_flag_rate)
    row = _locked_threshold_row(
        "future_test_posthoc_threshold_cap",
        y_arr,
        score_arr,
        amount,
        threshold,
        target_legit_flag_rate,
    )
    row["diagnostic_only"] = True
    row["may_underuse_budget_due_to_ties"] = True
    return row


def posthoc_budget_match(
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    target_legit_flag_rate: float = 0.001,
) -> dict:
    """Backward-compatible alias for the scalar threshold-cap diagnostic."""
    return posthoc_threshold_cap(y, score, amount, target_legit_flag_rate)


def ranked_capacity_metrics(
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    alerts_per_10k: float,
) -> dict:
    """Evaluate exact top-k routing under a total alert-capacity budget.

    The primary sort is descending model score. Equal scores are resolved by a stable non-label
    event key, so capacity is exactly enforceable without using fraud labels or amount as a tie-break.
    """
    if alerts_per_10k < 0:
        raise ValueError("alerts_per_10k must be non-negative")
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    if not (len(y_arr) == len(score_arr) == len(amount_arr) == len(key_arr)):
        raise ValueError("y, score, amount and event_key must have equal length")
    if len(y_arr) == 0:
        raise ValueError("Need at least one row")
    if np.any(~np.isfinite(score_arr)):
        raise ValueError("Scores must be finite")

    k = min(len(y_arr), int(np.floor(alerts_per_10k * len(y_arr) / 10_000)))
    pred = np.zeros(len(y_arr), dtype=bool)
    boundary_score = math.nan
    boundary_tie_n = 0
    boundary_tie_selected_n = 0
    if k > 0:
        order = np.lexsort((key_arr, -score_arr))
        selected = order[:k]
        pred[selected] = True
        boundary_score = float(score_arr[order[k - 1]])
        boundary_mask = score_arr == boundary_score
        boundary_tie_n = int(boundary_mask.sum())
        boundary_tie_selected_n = int((pred & boundary_mask).sum())

    fraud = y_arr == 1
    legit = ~fraud
    perf = rule_metrics(y_arr, pred, amount_arr)
    return {
        "target_alerts_per_10k": float(alerts_per_10k),
        "capacity_n": int(k),
        "alerts": int(pred.sum()),
        "actual_alerts_per_10k": float(pred.mean() * 10_000),
        "precision": perf["precision"],
        "recall": perf["recall"],
        "legit_flag_rate": perf["legit_flag_rate"],
        "legit_alerts_per_10k": float(perf["legit_flag_rate"] * 10_000),
        "fraud_value_recall": perf["fraud_value_recall"],
        "fraud_alerts": int((pred & fraud).sum()),
        "legit_alerts": int((pred & legit).sum()),
        "boundary_score": boundary_score,
        "boundary_tie_n": boundary_tie_n,
        "boundary_tie_selected_n": boundary_tie_selected_n,
        "tie_breaker": "stable_non_label_event_key",
    }


def ranked_capacity_frontier(
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    budgets_per_10k: Sequence[float] = (10, 25, 50, 100),
) -> pd.DataFrame:
    rows = [
        ranked_capacity_metrics(y, score, amount, event_key, budget)
        for budget in budgets_per_10k
    ]
    return pd.DataFrame(rows)


def ranked_capacity_windows(
    step: Sequence[int],
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    event_key: Sequence[int],
    alerts_per_10k: float = 50,
    n_windows: int = 3,
) -> pd.DataFrame:
    """Apply the same governed total alert capacity independently to contiguous future windows."""
    step_arr = np.asarray(step, dtype=int)
    unique_steps = np.unique(step_arr)
    if len(unique_steps) < n_windows:
        raise ValueError("Need at least one distinct step per capacity window")
    chunks = [c for c in np.array_split(unique_steps, n_windows) if len(c)]
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    amount_arr = np.asarray(amount, dtype=float)
    key_arr = np.asarray(event_key, dtype=np.uint64)
    rows: list[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        mask = np.isin(step_arr, chunk)
        row = ranked_capacity_metrics(
            y_arr[mask], score_arr[mask], amount_arr[mask], key_arr[mask], alerts_per_10k
        )
        row.update({
            "period": f"future_window_{idx}",
            "step_min": int(chunk.min()),
            "step_max": int(chunk.max()),
            "n": int(mask.sum()),
            "fraud_n": int(y_arr[mask].sum()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def recipient_intensity_score(df: pd.DataFrame) -> np.ndarray:
    """Label-free recipient activity score from strictly prior-step features."""
    missing = [c for c in RECIPIENT_SIGNAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing recipient signal columns: {missing}")
    fanin = np.log1p(np.clip(df["recipient_fanin_24h"].to_numpy(dtype=float), 0, None))
    tx7d = np.log1p(np.clip(df["recipient_tx_7d"].to_numpy(dtype=float), 0, None))
    amount24h = np.log1p(np.clip(df["recipient_amount_24h"].to_numpy(dtype=float), 0, None))
    senders7d = np.log1p(
        np.clip(df["recipient_unique_senders_7d"].to_numpy(dtype=float), 0, None)
    )
    return fanin + 0.5 * tx7d + 0.25 * amount24h + senders7d


def recipient_signal_audit(
    validation: pd.DataFrame,
    future: pd.DataFrame,
    target_legit_flag_rate: float = 0.001,
) -> pd.DataFrame:
    """Evaluate label-free recipient signals with thresholds chosen on validation only."""
    required = set(RECIPIENT_SIGNAL_COLUMNS + ["is_fraud", "amount"])
    for name, frame in (("validation", validation), ("future", future)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    signals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for column in RECIPIENT_SIGNAL_COLUMNS:
        signals.append(
            (
                column,
                validation[column].to_numpy(dtype=float),
                future[column].to_numpy(dtype=float),
            )
        )
    signals.append(
        (
            "recipient_intensity_score",
            recipient_intensity_score(validation),
            recipient_intensity_score(future),
        )
    )

    val_y = validation["is_fraud"].to_numpy(dtype=int)
    future_y = future["is_fraud"].to_numpy(dtype=int)
    future_amount = future["amount"].to_numpy(dtype=float)
    rows: list[dict] = []
    for name, val_score, future_score in signals:
        threshold = threshold_at_legit_rate(val_y, val_score, target_legit_flag_rate)
        val_pred = val_score >= threshold
        val_legit = val_y == 0
        pred = future_score >= threshold
        perf = rule_metrics(future_y, pred, future_amount)
        auc = float(roc_auc_score(future_y, future_score))
        fraud_values = future_score[future_y == 1]
        legit_values = future_score[future_y == 0]
        rows.append(
            {
                "signal": name,
                "validation_threshold": float(threshold),
                "target_validation_legit_flag_rate": float(target_legit_flag_rate),
                "validation_actual_legit_flag_rate": float(val_pred[val_legit].mean()),
                "future_auc": auc,
                **perf,
                "future_fraud_median": float(np.median(fraud_values)),
                "future_legit_median": float(np.median(legit_values)),
                "future_fraud_p90": float(np.quantile(fraud_values, 0.90)),
                "future_legit_p90": float(np.quantile(legit_values, 0.90)),
            }
        )
    return pd.DataFrame(rows)
