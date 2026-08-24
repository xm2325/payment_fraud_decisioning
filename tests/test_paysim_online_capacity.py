import numpy as np

from fraud_decisioning.paysim_online_capacity import (
    batch_vs_backlog_capacity,
    batch_vs_online_capacity,
    online_accrual_capacity_mask,
    online_backlog_capacity_mask,
)


def test_online_capacity_cannot_defer_early_slots_to_future_scores():
    step = np.repeat([1, 2], 10)
    score = np.r_[np.arange(10, dtype=float), np.arange(100, 110, dtype=float)]
    key = np.arange(20, dtype=np.uint64)

    selected, schedule = online_accrual_capacity_mask(
        step, score, key, alerts_per_10k=5000
    )

    # Ten final slots total. A retrospective whole-window top-k could spend all
    # ten on step 2, but the strict online contract must consume the five slots
    # already earned after step 1 using step-1 transactions only.
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

    for compare in (batch_vs_online_capacity, batch_vs_backlog_capacity):
        row, schedule = compare(
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


def test_batch_and_current_step_can_have_same_budget_but_different_queue():
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


def test_backlog_can_review_an_older_pending_transaction_later():
    step = np.repeat([1, 2], 4)
    score = np.array([10.0, 9.0, 1.0, 0.0, 2.0, 1.5, 1.0, 0.5])
    key = np.arange(8, dtype=np.uint64)

    selected, review_step, schedule = online_backlog_capacity_mask(
        step, score, key, alerts_per_10k=2500
    )

    # One slot is earned at each step. Step 1 takes score 10; at step 2 the
    # best seen-but-unreviewed item is the old score-9 transaction.
    assert selected.sum() == 2
    assert selected[0]
    assert selected[1]
    assert review_step[0] == 1
    assert review_step[1] == 2
    assert np.all(review_step[selected] >= step[selected])
    assert schedule.selected_cumulative.tolist() == [1, 2]
    assert schedule.pending_after_selection.tolist() == [3, 6]


def test_backlog_uses_stable_event_key_across_pending_ties():
    step = np.repeat([1, 2], 4)
    score = np.ones(8, dtype=float)
    key = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint64)

    selected, review_step, _ = online_backlog_capacity_mask(
        step, score, key, alerts_per_10k=2500
    )

    # Step 1 selects the smallest key available then; step 2 selects the new
    # smallest key from all seen pending rows.
    assert np.flatnonzero(selected).tolist() == [3, 7]
    assert review_step[3] == 1
    assert review_step[7] == 2


def test_backlog_never_uses_future_scores_but_can_reduce_strict_step_penalty():
    step = np.repeat([1, 2], 4)
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=int)
    amount = np.ones(8, dtype=float)
    score = np.array([10.0, 9.0, 1.0, 0.0, 2.0, 1.5, 1.0, 0.5])
    key = np.arange(8, dtype=np.uint64)

    strict, _ = batch_vs_online_capacity(
        step, y, score, amount, key, alerts_per_10k=2500
    )
    backlog, _ = batch_vs_backlog_capacity(
        step, y, score, amount, key, alerts_per_10k=2500
    )

    assert strict["online_fraud_recall"] == 0.5
    assert backlog["online_fraud_recall"] == 1.0
    assert backlog["queue_overlap_rate"] == 1.0
    assert backlog["review_delay_max_steps"] == 1
