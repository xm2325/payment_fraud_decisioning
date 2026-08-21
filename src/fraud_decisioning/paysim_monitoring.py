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
    """Measure budget drift after locking a validation-selected threshold."""
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
    """Evaluate a locked threshold across contiguous future step windows."""
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


def posthoc_budget_match(
    y: Sequence[int],
    score: Sequence[float],
    amount: Sequence[float],
    target_legit_flag_rate: float = 0.001,
) -> dict:
    """Post-hoc oracle diagnostic: threshold required to restore the target budget."""
    y_arr = np.asarray(y, dtype=int)
    score_arr = np.asarray(score, dtype=float)
    threshold = threshold_at_legit_rate(y_arr, score_arr, target_legit_flag_rate)
    row = _locked_threshold_row(
        "future_test_posthoc_budget_match",
        y_arr,
        score_arr,
        amount,
        threshold,
        target_legit_flag_rate,
    )
    row["diagnostic_only"] = True
    return row


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
                "future_auc": auc,
                **perf,
                "future_fraud_median": float(np.median(fraud_values)),
                "future_legit_median": float(np.median(legit_values)),
                "future_fraud_p90": float(np.quantile(fraud_values, 0.90)),
                "future_legit_p90": float(np.quantile(legit_values, 0.90)),
            }
        )
    return pd.DataFrame(rows)
