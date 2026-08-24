import numpy as np

from fraud_decisioning.paysim_policy_promotion import (
    exact_capacity_mask,
    paired_circular_block_bootstrap,
    promotion_decision,
)


def test_exact_capacity_mask_uses_stable_event_key_for_ties():
    score = np.ones(20)
    key = np.arange(20, 0, -1, dtype=np.uint64)
    selected = exact_capacity_mask(score, key, alerts_per_10k=1000)
    assert selected.sum() == 2
    assert np.flatnonzero(selected).tolist() == [18, 19]


def _paired_fixture():
    n_steps = 10
    rows_per_step = 20
    n = n_steps * rows_per_step
    step = np.repeat(np.arange(1, n_steps + 1), rows_per_step)
    y = np.zeros(n, dtype=int)
    amount = np.ones(n, dtype=float)
    incumbent_score = np.zeros(n, dtype=float)
    candidate_score = np.zeros(n, dtype=float)

    for s in range(n_steps):
        base = s * rows_per_step
        low_fraud = base
        high_fraud = base + 1
        y[[low_fraud, high_fraud]] = 1
        amount[low_fraud] = 1.0
        amount[high_fraud] = 100.0
        incumbent_score[low_fraud] = 2.0
        candidate_score[high_fraud] = 2.0

    event_key = np.arange(n, dtype=np.uint64)
    return step, y, amount, event_key, incumbent_score, candidate_score


def test_paired_block_bootstrap_detects_stable_value_gain_without_case_tradeoff():
    args = _paired_fixture()
    result = paired_circular_block_bootstrap(
        *args,
        alerts_per_10k=500,
        block_steps=2,
        n_bootstrap=500,
        tail_alpha=0.025,
        seed=7,
    )
    intervals = result["delta_intervals"]
    assert intervals["fraud_value_recall"]["point_delta"] > 0.9
    assert intervals["fraud_value_recall"]["lower"] > 0
    assert abs(intervals["fraud_recall"]["point_delta"]) < 1e-12
    assert abs(intervals["precision"]["point_delta"]) < 1e-12

    decision = promotion_decision(
        "value_first",
        incumbent_alpha=0.25,
        candidate_alpha=1.0,
        uncertainty=result,
    )
    assert decision["decision"] == "PROMOTE"


def test_promotion_gate_rejects_negative_primary_result_and_skips_same_policy():
    args = _paired_fixture()
    step, y, amount, key, incumbent_score, candidate_score = args
    result = paired_circular_block_bootstrap(
        step,
        y,
        amount,
        key,
        candidate_score,
        incumbent_score,
        alerts_per_10k=500,
        block_steps=2,
        n_bootstrap=500,
        tail_alpha=0.025,
        seed=11,
    )
    rejected = promotion_decision(
        "value_first",
        incumbent_alpha=1.0,
        candidate_alpha=0.25,
        uncertainty=result,
    )
    assert rejected["decision"] == "KEEP_INCUMBENT"

    unchanged = promotion_decision(
        "balanced",
        incumbent_alpha=0.25,
        candidate_alpha=0.25,
        uncertainty=None,
    )
    assert unchanged["decision"] == "NO_POLICY_CHANGE"
