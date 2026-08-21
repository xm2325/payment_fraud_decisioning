import numpy as np
import pandas as pd

from fraud_decisioning.paysim_monitoring import (
    future_budget_windows,
    locked_threshold_drift,
    recipient_intensity_score,
    recipient_signal_audit,
)


def test_locked_threshold_drift_reports_budget_multiplier():
    val_y = np.array([0, 0, 0, 0, 1])
    val_score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    val_amount = np.array([1, 1, 1, 1, 10.0])
    future_y = np.array([0, 0, 0, 0, 1])
    future_score = np.array([0.1, 0.7, 0.8, 0.9, 0.95])
    future_amount = np.array([1, 1, 1, 1, 10.0])
    out = locked_threshold_drift(
        val_y, val_score, val_amount,
        future_y, future_score, future_amount,
        threshold=0.5,
        target_legit_flag_rate=0.25,
    )
    assert out.loc[out.period == "validation", "legit_flag_rate"].iloc[0] == 0.0
    assert out.loc[out.period == "future_test", "legit_flag_rate"].iloc[0] == 0.75
    assert out.loc[out.period == "future_test", "budget_multiplier_vs_target"].iloc[0] == 3.0


def test_future_budget_windows_are_contiguous():
    step = np.arange(1, 7)
    y = np.array([0, 0, 0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8])
    amount = np.ones(6)
    out = future_budget_windows(step, y, score, amount, threshold=0.5, n_windows=3)
    assert out[["step_min", "step_max"]].values.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_recipient_intensity_score_is_monotone_for_higher_activity():
    df = pd.DataFrame({
        "recipient_fanin_24h": [1.0, 10.0],
        "recipient_tx_7d": [2.0, 20.0],
        "recipient_amount_24h": [100.0, 10000.0],
        "recipient_unique_senders_7d": [1.0, 8.0],
    })
    score = recipient_intensity_score(df)
    assert score[1] > score[0]


def test_recipient_signal_thresholds_are_validation_selected():
    val = pd.DataFrame({
        "recipient_fanin_24h": [0, 1, 2, 3, 10],
        "recipient_tx_7d": [0, 1, 2, 3, 10],
        "recipient_amount_24h": [0, 10, 20, 30, 100],
        "recipient_unique_senders_7d": [0, 1, 2, 3, 10],
        "is_fraud": [0, 0, 0, 0, 1],
        "amount": [1, 1, 1, 1, 10],
    })
    future = val.copy()
    out = recipient_signal_audit(val, future, target_legit_flag_rate=0.25)
    assert set(out.signal) == {
        "recipient_fanin_24h",
        "recipient_tx_7d",
        "recipient_amount_24h",
        "recipient_unique_senders_7d",
        "recipient_intensity_score",
    }
    assert (out.target_validation_legit_flag_rate == 0.25).all()
