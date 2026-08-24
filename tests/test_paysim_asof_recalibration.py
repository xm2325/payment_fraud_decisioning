import numpy as np
import pandas as pd

from fraud_decisioning.paysim_asof_recalibration import (
    contiguous_future_windows,
    expanding_recalibration_mask,
    matured_incremental_mask,
    method_summary,
    recalibration_metric_row,
)


def test_future_windows_are_contiguous_and_cover_steps():
    step = np.repeat(np.arange(595, 744), 2)
    windows = contiguous_future_windows(step, n_windows=3)
    assert [(w.step_min, w.step_max) for w in windows] == [(595, 644), (645, 694), (695, 743)]


def test_matured_labels_respect_lag_and_initial_boundary():
    step = np.arange(510, 700)
    mask24, cutoff24 = matured_incremental_mask(
        step, window_start_step=645, maturity_lag_steps=24, initial_calibration_step_max=519
    )
    mask168, cutoff168 = matured_incremental_mask(
        step, window_start_step=645, maturity_lag_steps=168, initial_calibration_step_max=519
    )
    assert cutoff24 == 620
    assert cutoff168 == 476
    assert step[mask24].min() == 520 and step[mask24].max() == 620
    assert not mask168.any()


def test_expanding_mask_always_keeps_initial_approved_calibration_rows():
    step = np.arange(440, 650)
    mask, cutoff = expanding_recalibration_mask(
        step,
        window_start_step=595,
        maturity_lag_steps=168,
        initial_calibration_step_min=446,
        initial_calibration_step_max=519,
    )
    selected = step[mask]
    assert cutoff == 426
    assert selected.min() == 446
    assert selected.max() == 519


def test_shorter_lag_never_has_fewer_refresh_rows():
    step = np.arange(446, 695)
    counts = []
    for lag in (168, 24, 0):
        mask, _ = expanding_recalibration_mask(
            step,
            window_start_step=695,
            maturity_lag_steps=lag,
            initial_calibration_step_min=446,
            initial_calibration_step_max=519,
        )
        counts.append(int(mask.sum()))
    assert counts[0] <= counts[1] <= counts[2]


def test_recalibration_metrics_and_summary_are_finite():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.20, 0.70, 0.90])
    window = contiguous_future_windows(np.arange(10, 16), n_windows=3)[0]
    row = recalibration_metric_row(
        method="asof_24h", window=window, y=y, probability=p,
        calibration_n=100, calibration_max_step=8, maturity_lag_steps=24,
    )
    frame = pd.DataFrame([row, {**row, "window": "future_window_2", "brier": row["brier"] * 1.1}])
    summary = method_summary(frame)
    assert row["calibration_n"] == 100
    assert 0 <= row["brier"] <= 1
    assert np.isfinite(summary.loc[0, "mean_brier"])


def test_negative_label_lag_is_rejected():
    try:
        matured_incremental_mask(
            np.arange(10), window_start_step=10, maturity_lag_steps=-1,
            initial_calibration_step_max=5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative maturity lag should fail")
