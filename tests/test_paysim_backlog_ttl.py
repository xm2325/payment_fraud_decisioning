import numpy as np

from fraud_decisioning.paysim_backlog_handoff import continuous_backlog_handoff
from fraud_decisioning.paysim_backlog_ttl import (
    continuous_rescore_backlog_ttl,
    expiry_metrics,
    ttl_label,
)


def _fixture():
    step = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
    key = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.uint64)
    regime_by_step = {1: 1, 2: 1, 3: 2, 4: 2}
    score_by_regime = {
        1: np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        2: np.array([0.1, 0.2, 0.3, 0.4, 0.95, 0.85, 0.75, 0.65]),
    }
    return step, key, regime_by_step, score_by_regime


def test_infinite_ttl_reproduces_rescore_pending_exactly():
    step, key, regime_by_step, score_by_regime = _fixture()
    expected = continuous_backlog_handoff(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        handoff_mode="rescore_pending",
    )
    actual = continuous_rescore_backlog_ttl(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        max_age_steps=None,
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert not actual[3].any()


def test_zero_ttl_never_reviews_an_older_step():
    step, key, regime_by_step, score_by_regime = _fixture()
    selected, review_step, _, expired, _, _ = continuous_rescore_backlog_ttl(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=5000,
        max_age_steps=0,
    )
    assert np.all(review_step[selected] == step[selected])
    assert expired.any()


def test_finite_ttl_caps_review_delay_and_reduces_refresh_rescore_pool():
    step, key, regime_by_step, score_by_regime = _fixture()
    finite = continuous_rescore_backlog_ttl(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=2500,
        max_age_steps=1,
    )
    infinite = continuous_rescore_backlog_ttl(
        step,
        key,
        regime_by_step,
        score_by_regime,
        alerts_per_10k=2500,
        max_age_steps=None,
    )
    selected, review_step, _, _, _, finite_refresh = finite
    assert np.all((review_step[selected] - step[selected]) <= 1)
    assert int(finite_refresh.iloc[0].pending_rescored) <= int(
        infinite[5].iloc[0].pending_rescored
    )


def test_expiry_diagnostics_are_label_based_reporting_only():
    y = np.array([0, 1, 1, 0])
    amount = np.array([1.0, 10.0, 30.0, 2.0])
    expired = np.array([True, True, False, False])
    row = expiry_metrics(y, amount, expired)
    assert row["expired_cases"] == 2
    assert row["expired_fraud_cases"] == 1
    assert row["expired_fraud_case_share"] == 0.5
    assert row["expired_fraud_value_share"] == 0.25


def test_ttl_label_validates_negative_values():
    assert ttl_label(None) == "infinite"
    assert ttl_label(20) == "20"
    try:
        ttl_label(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative TTL should be rejected")
