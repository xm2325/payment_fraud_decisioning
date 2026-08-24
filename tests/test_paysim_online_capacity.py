import numpy as np

from fraud_decisioning.paysim_online_capacity import (
    batch_vs_online_capacity,
    online_accrual_capacity_mask,
)


def test_online_capacity_cannot_defer_early_slots_to_future_scores():
    step = np.repeat([1, 2], 10)
    score = np.r_[np.arange(10, dtype=float), np.arange(100, 110, dtype=float)]
    key = np.arange(20, dtype=np.uint64)

    selected, schedule = online_accrual_capacity_mask(
        step, score, key, alerts_per_10k=5000
    )

    # Ten final slots total. A retrospective whole-window top-k could spend all
    # ten on step 2, but the online contract must consume the five slots already
    # earned after step 1 using step-1 transactions only.
    assert selected[:10].sum() == 5
    assert selected[10:].sum() == 5
    assert selected.sum() == 10
    assert schedule.selected_cumulative.tolist() == [5, 10]


def test_fractional_capacity_accrues_without_losing_final_budget():
    step = np.repeat(np.arange(1, 5), 3)
    score = np.arange(12, dtype=float)
    key = np.arange(12, dtype=np.uint64)

    selected, schedule = online_accrual_capacity_mask(
        step, score, key, alerts_per_10k=2000
    )

    # floor(0.2 * 12) = 2 final slots. The cumulative floor carries fractional
    # entitlement until a whole review slot becomes available.
    assert selected.sum() == 2
    assert schedule.cumulative_entitlement.tolist() == [0, 1, 1, 2]
    assert schedule.selected_cumulative.tolist() == [0, 1, 1, 2]


def test_ties_use_stable_non_label_event_key_within_step():
    step = np.ones(10, dtype=int)
    score = np.ones(10, dtype=float)
    key = np.arange(10, 0, -1, dtype=np.uint64)

    selected, _ = online_accrual_capacity_mask(
        step, score, key, alerts_per_10k=2000
    )
    assert selected.sum() == 2
    assert np.flatnonzero(selected).tolist() == [8, 9]


def test_one_step_online_matches_batch_exact_capacity():
    n = 100
    step = np.ones(n, dtype=int)
    y = np.zeros(n, dtype=int)
    y[[1, 10, 50]] = 1
    amount = np.linspace(1, 100, n)
    score = np.linspace(0, 1, n)
    key = np.arange(n, dtype=np.uint64)

    row, schedule = batch_vs_online_capacity(
        step,
        y,
        score,
        amount,
        key,
        alerts_per_10k=1000,
    )
    assert row["batch_alerts"] == row["online_alerts"] == 10
    assert row["queue_overlap_rate"] == 1.0
    assert row["replacement_count"] == 0
    assert row["delta_precision"] == 0.0
    assert row["delta_fraud_recall"] == 0.0
    assert row["delta_fraud_value_recall"] == 0.0
    assert schedule.selected_cumulative.iloc[-1] == 10


def test_batch_and_online_can_have_same_budget_but_different_queue():
    step = np.repeat([1, 2], 10)
    y = np.zeros(20, dtype=int)
    y[[9, 18, 19]] = 1
    amount = np.ones(20, dtype=float)
    score = np.r_[np.arange(10, dtype=float), np.arange(100, 110, dtype=float)]
    key = np.arange(20, dtype=np.uint64)

    row, _ = batch_vs_online_capacity(
        step,
        y,
        score,
        amount,
        key,
        alerts_per_10k=5000,
    )
    assert row["batch_alerts"] == row["online_alerts"] == 10
    assert row["replacement_count"] == 5
    assert row["queue_overlap_rate"] == 0.5
    assert row["queue_jaccard"] == 1 / 3
