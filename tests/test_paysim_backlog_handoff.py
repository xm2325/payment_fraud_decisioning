import numpy as np

from fraud_decisioning.paysim_backlog_handoff import (
    continuous_backlog_handoff,
    queue_metrics,
    queue_overlap,
)


def _toy():
    step = np.repeat(np.arange(1, 7), 4)
    key = np.arange(len(step), dtype=np.uint64)
    regime_by_step = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
    score1 = np.linspace(0.0, 1.0, len(step))
    score2 = score1.copy()
    score3 = score1.copy()
    # New regimes strongly prefer some older pending cases.
    score2[:8] += 5.0
    score3[:16] += 10.0
    scores = {1: score1, 2: score2, 3: score3}
    return step, key, regime_by_step, scores


def test_all_handoff_modes_consume_same_continuous_capacity():
    step, key, regimes, scores = _toy()
    masks = {}
    for mode in ("retain_old_scores", "rescore_pending", "drop_pending"):
        selected, review_step, review_regime, schedule, refresh = continuous_backlog_handoff(
            step,
            key,
            regimes,
            scores,
            alerts_per_10k=2500,
            handoff_mode=mode,
        )
        masks[mode] = selected
        assert selected.sum() == 6
        assert schedule.selected_cumulative.iloc[-1] == 6
        assert np.all(review_step[selected] >= step[selected])
        assert np.all(review_regime[selected] >= 1)
        assert refresh.refresh_step.tolist() == [3, 5]
    assert masks["retain_old_scores"].sum() == masks["rescore_pending"].sum()
    assert masks["retain_old_scores"].sum() == masks["drop_pending"].sum()


def test_rescore_pending_records_rescored_backlog():
    step, key, regimes, scores = _toy()
    _, _, _, _, refresh = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="rescore_pending",
    )
    assert (refresh.rescored_pending > 0).all()
    assert (refresh.dropped_pending == 0).all()


def test_drop_pending_discards_old_unreviewed_cases_at_refresh():
    step, key, regimes, scores = _toy()
    _, _, _, _, refresh = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="drop_pending",
    )
    assert (refresh.dropped_pending == refresh.pending_before).all()
    assert (refresh.pending_after_handoff == 0).all()


def test_retain_mode_never_changes_pending_scores_at_refresh():
    step, key, regimes, scores = _toy()
    retain, _, _, _, refresh = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="retain_old_scores",
    )
    rescore, _, _, _, _ = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="rescore_pending",
    )
    assert (refresh.rescored_pending == 0).all()
    # The crafted score changes should make at least one handoff assignment differ.
    assert not np.array_equal(retain, rescore)


def test_metrics_and_overlap_are_consistent():
    step, key, regimes, scores = _toy()
    left, left_review, _, _, _ = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="retain_old_scores",
    )
    right, right_review, _, _, _ = continuous_backlog_handoff(
        step,
        key,
        regimes,
        scores,
        alerts_per_10k=2500,
        handoff_mode="rescore_pending",
    )
    y = np.zeros(len(step), dtype=int)
    y[[3, 7, 11, 15, 19, 23]] = 1
    amount = np.arange(1, len(step) + 1, dtype=float)
    left_metrics = queue_metrics(step, y, amount, left, left_review)
    right_metrics = queue_metrics(step, y, amount, right, right_review)
    overlap = queue_overlap(left, right)
    assert left_metrics["alerts"] == right_metrics["alerts"] == 6
    assert 0 <= overlap["jaccard"] <= 1
    assert overlap["left_alerts"] == overlap["right_alerts"] == 6


def test_unknown_mode_is_rejected():
    step, key, regimes, scores = _toy()
    try:
        continuous_backlog_handoff(
            step,
            key,
            regimes,
            scores,
            alerts_per_10k=2500,
            handoff_mode="future_magic",
        )
    except ValueError as exc:
        assert "handoff_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
