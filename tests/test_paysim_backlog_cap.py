import numpy as np

from fraud_decisioning.paysim_backlog_cap import (
    cap_label,
    continuous_rescore_bounded_backlog,
    eviction_metrics,
)
from fraud_decisioning.paysim_backlog_handoff import continuous_backlog_handoff


def _fixture():
    step = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
    key = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint64)
    regime_by_step = {1: 1, 2: 1, 3: 2, 4: 2}
    score_by_regime = {
        1: np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        2: np.array([0.1, 0.2, 0.3, 0.4, 0.95, 0.85, 0.75, 0.65]),
    }
    return step, key, regime_by_step, score_by_regime


def test_infinite_cap_reproduces_rescore_pending_exactly():
    step, key, regime_by_step, score_by_regime = _fixture()
    expected = continuous_backlog_handoff(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        handoff_mode="rescore_pending",
    )
    actual = continuous_rescore_bounded_backlog(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        max_pending_cases=None,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert not actual[3].any()


def test_zero_cap_never_carries_pending_cases_forward():
    step, key, regime_by_step, score_by_regime = _fixture()
    selected, review_step, _, evicted, schedule, refresh = continuous_rescore_bounded_backlog(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        max_pending_cases=0,
    )
    assert np.all(review_step[selected] == step[selected])
    assert (schedule.pending_after_selection_and_cap == 0).all()
    assert int(refresh.pending_rescored.sum()) == 0
    assert evicted.any()


def test_cap_is_respected_at_every_step_and_refresh():
    step, key, regime_by_step, score_by_regime = _fixture()
    _, _, _, _, schedule, refresh = continuous_rescore_bounded_backlog(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=2500,
        max_pending_cases=2,
    )
    assert int(schedule.pending_after_selection_and_cap.max()) <= 2
    assert int(refresh.pending_rescored.max()) <= 2


def test_bounded_pool_evicts_lowest_priority_case():
    step = np.array([1, 1, 1, 2], dtype=int)
    key = np.array([10, 11, 12, 13], dtype=np.uint64)
    regime_by_step = {1: 1, 2: 1}
    scores = {1: np.array([0.9, 0.1, 0.8, 0.7])}
    selected, _, _, evicted, schedule, _ = continuous_rescore_bounded_backlog(
        step,
        key,
        regime_by_step,
        scores,
        alerts_per_10k=2500,
        max_pending_cases=1,
    )
    # Step 1 earns no review slot. The one-case retained pool must keep 0.9.
    assert evicted[1]
    assert evicted[2]
    assert not evicted[0]
    assert int(schedule.iloc[0].pending_after_selection_and_cap) == 1
    # Step 2 earns one slot and should eventually review the retained 0.9 case.
    assert selected[0]


def test_eviction_diagnostics_are_reporting_only():
    y = np.array([0, 1, 1, 0])
    amount = np.array([1.0, 10.0, 30.0, 2.0])
    evicted = np.array([True, True, False, False])
    row = eviction_metrics(y, amount, evicted)
    assert row["evicted_cases"] == 2
    assert row["evicted_fraud_cases"] == 1
    assert row["evicted_fraud_case_share"] == 0.5
    assert row["evicted_fraud_value_share"] == 0.25


def test_cap_label_rejects_negative_values():
    assert cap_label(None) == "infinite"
    assert cap_label(25000) == "25000"
    try:
        cap_label(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative cap should be rejected")
