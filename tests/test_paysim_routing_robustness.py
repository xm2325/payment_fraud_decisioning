import numpy as np
import pandas as pd

from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)


def test_validation_window_grid_uses_fixed_validation_scale_and_all_windows():
    step = np.repeat(np.arange(1, 7), 4)
    y = np.tile(np.array([1, 0, 0, 0]), 6)
    p = np.linspace(0.95, 0.05, len(step))
    amount = np.arange(1, len(step) + 1, dtype=float) * 10
    key = np.arange(len(step), dtype=np.uint64)

    grid = validation_window_alpha_grid(
        step, y, p, amount, key,
        alerts_per_10k=2500,
        n_windows=3,
        alphas=(0.0, 0.5, 1.0),
    )
    assert set(grid.window) == {
        "validation_window_1", "validation_window_2", "validation_window_3"
    }
    assert set(grid.alpha) == {0.0, 0.5, 1.0}
    assert grid.amount_scale.nunique() == 1


def test_robustness_summary_uses_worst_and_mean_window_metrics():
    frame = pd.DataFrame([
        {"alpha": 0.0, "amount_scale": 10.0, "recall": 0.8, "precision": 0.5,
         "fraud_value_recall": 0.6, "balanced_hmean": 0.6857},
        {"alpha": 0.0, "amount_scale": 10.0, "recall": 0.4, "precision": 0.7,
         "fraud_value_recall": 0.9, "balanced_hmean": 0.5538},
        {"alpha": 0.5, "amount_scale": 10.0, "recall": 0.6, "precision": 0.6,
         "fraud_value_recall": 0.7, "balanced_hmean": 0.6462},
        {"alpha": 0.5, "amount_scale": 10.0, "recall": 0.6, "precision": 0.6,
         "fraud_value_recall": 0.7, "balanced_hmean": 0.6462},
    ])
    summary = robustness_summary(frame)
    row0 = summary.loc[summary.alpha == 0.0].iloc[0]
    row5 = summary.loc[summary.alpha == 0.5].iloc[0]
    assert row0.min_recall == 0.4
    assert row0.mean_recall == 0.6
    assert row5.min_recall == 0.6
    assert row5.recall_range == 0.0


def test_robust_profile_selection_prefers_worst_window_stability():
    summary = pd.DataFrame([
        {"alpha": 0.0, "amount_scale": 10.0, "min_recall": 0.30, "mean_recall": 0.70,
         "min_precision": 0.40, "mean_precision": 0.60,
         "min_fraud_value_recall": 0.30, "mean_fraud_value_recall": 0.80,
         "min_balanced_hmean": 0.30, "mean_balanced_hmean": 0.74,
         "recall_range": 0.8, "value_recall_range": 0.7},
        {"alpha": 0.5, "amount_scale": 10.0, "min_recall": 0.55, "mean_recall": 0.60,
         "min_precision": 0.50, "mean_precision": 0.55,
         "min_fraud_value_recall": 0.60, "mean_fraud_value_recall": 0.65,
         "min_balanced_hmean": 0.57, "mean_balanced_hmean": 0.62,
         "recall_range": 0.1, "value_recall_range": 0.1},
        {"alpha": 1.0, "amount_scale": 10.0, "min_recall": 0.20, "mean_recall": 0.40,
         "min_precision": 0.45, "mean_precision": 0.50,
         "min_fraud_value_recall": 0.70, "mean_fraud_value_recall": 0.75,
         "min_balanced_hmean": 0.31, "mean_balanced_hmean": 0.50,
         "recall_range": 0.4, "value_recall_range": 0.1},
    ])
    selected = select_robust_profiles(summary)
    assert selected.loc[selected.profile == "case_first", "alpha"].item() == 0.5
    assert selected.loc[selected.profile == "balanced", "alpha"].item() == 0.5
    assert selected.loc[selected.profile == "value_first", "alpha"].item() == 1.0
    assert set(selected.selection_split) == {"validation_windows_only"}
