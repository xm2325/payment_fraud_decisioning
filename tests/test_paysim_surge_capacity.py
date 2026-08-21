import numpy as np

from fraud_decisioning.paysim_surge_capacity import (
    capacity_from_score_load,
    evaluate_surge_windows,
    surge_trigger_sensitivity,
    validation_tail_reference,
)


def test_validation_tail_reference_reports_actual_tied_tail_rate():
    p = np.array([0.1, 0.2, 0.8, 0.8, 0.8, 0.9])
    threshold, rate = validation_tail_reference(p, tail_quantile=0.5)
    assert threshold == 0.8
    assert np.isclose(rate, 4 / 6)


def test_capacity_trigger_uses_scores_only_and_switches_at_multiplier():
    decision = capacity_from_score_load(
        [0.1, 0.9, 0.95, 0.99],
        tail_threshold=0.9,
        validation_tail_rate=0.25,
        baseline_alerts_per_10k=50,
        surge_alerts_per_10k=100,
        trigger_multiplier=1.5,
    )
    assert decision["surge_triggered"] is True
    assert decision["selected_alerts_per_10k"] == 100
    assert np.isclose(decision["score_tail_multiplier_vs_validation"], 3.0)


def test_surge_window_doubles_capacity_without_using_labels_for_trigger():
    step = np.array([1] * 10 + [2] * 10)
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    p = np.array([0.95, 0.9, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.99] * 8 + [0.1, 0.1])
    priority = p.copy()
    amount = np.ones(20)
    key = np.arange(20, dtype=np.uint64)
    out = evaluate_surge_windows(
        step, y, p, priority, amount, key,
        tail_threshold=0.9,
        validation_tail_rate=0.3,
        baseline_alerts_per_10k=1000,
        surge_alerts_per_10k=2000,
        trigger_multiplier=1.5,
        n_windows=2,
    )
    w1 = out[(out.period == "future_window_1") & (out.policy == "score_tail_surge")].iloc[0]
    w2 = out[(out.period == "future_window_2") & (out.policy == "score_tail_surge")].iloc[0]
    assert w1.selected_alerts_per_10k == 1000
    assert w2.selected_alerts_per_10k == 2000
    assert w2.alerts > w1.alerts


def test_trigger_sensitivity_does_not_pick_a_future_optimum():
    step = np.array([1] * 10 + [2] * 10)
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 2)
    p = np.array([0.95, 0.9, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [0.99] * 8 + [0.1, 0.1])
    amount = np.ones(20)
    key = np.arange(20, dtype=np.uint64)
    out = surge_trigger_sensitivity(
        step,
        y,
        p,
        p,
        amount,
        key,
        tail_threshold=0.9,
        validation_tail_rate=0.3,
        baseline_alerts_per_10k=1000,
        surge_alerts_per_10k=2000,
        trigger_multipliers=(1.5, 2.0, 3.0),
        n_windows=2,
    )
    assert set(out.trigger_multiplier) == {1.5, 2.0, 3.0}
    assert len(out) == 6
    # Window 1 has 30% tail load -> 1.0x, so it never surges.
    assert not out.loc[out.period == "future_window_1", "surge_triggered"].any()
    # Window 2 has 80% tail load -> 2.67x: it surges for 1.5/2.0 but not 3.0.
    window2 = out[out.period == "future_window_2"].set_index("trigger_multiplier")
    assert bool(window2.loc[1.5, "surge_triggered"])
    assert bool(window2.loc[2.0, "surge_triggered"])
    assert not bool(window2.loc[3.0, "surge_triggered"])
