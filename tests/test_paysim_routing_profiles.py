import numpy as np

from fraud_decisioning.paysim_routing_profiles import (
    alpha_grid_metrics,
    amount_scale_from_validation,
    evaluate_selected_profiles,
    priority_score,
    select_profiles,
)


def test_priority_score_endpoints_match_probability_and_expected_loss_order():
    p = np.array([0.2, 0.4, 0.1])
    amount = np.array([100.0, 10.0, 1000.0])
    scale = 100.0
    np.testing.assert_allclose(priority_score(p, amount, 0.0, scale), p)
    np.testing.assert_allclose(priority_score(p, amount, 1.0, scale), p * amount / scale)


def test_amount_scale_uses_positive_validation_median():
    assert amount_scale_from_validation([0.0, 10.0, 20.0, 100.0]) == 20.0
    assert amount_scale_from_validation([0.0, 0.0]) == 1.0


def test_profile_selection_is_driven_only_by_validation_grid_objectives():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
    amount = np.array([10, 1000, 500, 400, 2000, 50, 40, 5, 30, 20], dtype=float)
    key = np.arange(len(y), dtype=np.uint64)
    grid = alpha_grid_metrics(
        y, p, amount, key,
        alerts_per_10k=3000,
        alphas=(0.0, 0.5, 1.0),
    )
    selected = select_profiles(grid)
    assert set(selected.profile) == {"case_first", "balanced", "value_first"}
    assert set(selected.alpha).issubset({0.0, 0.5, 1.0})

    case_row = selected.loc[selected.profile == "case_first"].iloc[0]
    value_row = selected.loc[selected.profile == "value_first"].iloc[0]
    assert case_row.recall == grid.recall.max()
    assert value_row.fraud_value_recall == grid.fraud_value_recall.max()


def test_selected_profiles_keep_validation_alpha_on_future_data():
    val_y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    val_p = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    val_amount = np.array([10, 1000, 500, 50, 40, 2000, 20, 10], dtype=float)
    val_key = np.arange(len(val_y), dtype=np.uint64)
    grid = alpha_grid_metrics(
        val_y, val_p, val_amount, val_key,
        alerts_per_10k=2500,
        alphas=(0.0, 0.5, 1.0),
    )
    selected = select_profiles(grid)

    future_y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    future_p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    future_amount = np.array([5, 100, 5000, 20, 50, 40, 30, 1000], dtype=float)
    future_key = np.arange(100, 108, dtype=np.uint64)
    evaluated = evaluate_selected_profiles(
        selected, future_y, future_p, future_amount, future_key,
        budgets_per_10k=(2500,),
    )
    merged = evaluated.merge(selected[["profile", "alpha"]], on="profile", suffixes=("_future", "_selected"))
    np.testing.assert_allclose(merged.alpha_future, merged.alpha_selected)
