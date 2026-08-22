import numpy as np

from fraud_decisioning.paysim_nested_validation import calibration_metrics, nested_policy_cut


def test_nested_policy_cut_uses_predefined_validation_midpoint():
    assert nested_policy_cut(446, 595) == 520
    assert nested_policy_cut(10, 20) == 15


def test_nested_policy_cut_requires_room_for_both_periods():
    try:
        nested_policy_cut(10, 11)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_calibration_metrics_match_simple_well_calibrated_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = calibration_metrics(y, p, n_bins=5)
    assert metrics["n"] == 4
    assert metrics["fraud_n"] == 2
    assert np.isclose(metrics["fraud_rate"], 0.5)
    assert np.isclose(metrics["mean_predicted_risk"], 0.5)
    assert np.isclose(metrics["brier"], 0.0)
    assert np.isclose(metrics["ece_10bin"], 0.0)
    assert np.isclose(metrics["pr_auc"], 1.0)


def test_calibration_metrics_reject_unbounded_scores():
    try:
        calibration_metrics([0, 1], [0.1, 1.2])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
