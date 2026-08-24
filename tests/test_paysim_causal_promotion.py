import numpy as np

from fraud_decisioning.paysim_causal_promotion import (
    paired_causal_backlog_block_bootstrap,
)


def _toy_inputs():
    step = np.repeat(np.arange(1, 13), 20)
    n = len(step)
    y = np.zeros(n, dtype=int)
    y[[19, 39, 59, 99, 119, 139, 179, 199, 219]] = 1
    amount = np.linspace(1.0, 500.0, n)
    key = np.arange(n, dtype=np.uint64)
    incumbent = np.linspace(0.0, 1.0, n)
    candidate = incumbent.copy()
    candidate[y == 1] += 0.2
    return step, y, amount, key, incumbent, candidate


def test_causal_bootstrap_uses_equal_capacity_and_forward_delays():
    step, y, amount, key, incumbent, candidate = _toy_inputs()
    result = paired_causal_backlog_block_bootstrap(
        step,
        y,
        amount,
        key,
        incumbent,
        candidate,
        alerts_per_10k=1000,
        block_steps=3,
        n_bootstrap=200,
        tail_alpha=0.05,
        seed=7,
    )
    assert result["routing_contract"] == "seen_so_far_backlog"
    assert result["incumbent"]["alerts"] == result["candidate"]["alerts"] == 24
    assert result["incumbent"]["review_delay_mean_steps"] >= 0
    assert result["candidate"]["review_delay_mean_steps"] >= 0
    assert result["n_bootstrap_valid"] == 200


def test_identical_scores_give_zero_deltas_and_full_overlap():
    step, y, amount, key, incumbent, _ = _toy_inputs()
    result = paired_causal_backlog_block_bootstrap(
        step,
        y,
        amount,
        key,
        incumbent,
        incumbent,
        alerts_per_10k=1000,
        block_steps=3,
        n_bootstrap=200,
        tail_alpha=0.05,
        seed=11,
    )
    assert result["queue_overlap_rate"] == 1.0
    assert result["replacement_count"] == 0
    for metric in ("precision", "fraud_recall", "fraud_value_recall", "balanced_hmean"):
        interval = result["delta_intervals"][metric]
        assert interval["point_delta"] == 0.0
        assert interval["lower"] == 0.0
        assert interval["upper"] == 0.0


def test_invalid_block_length_is_rejected():
    step, y, amount, key, incumbent, candidate = _toy_inputs()
    try:
        paired_causal_backlog_block_bootstrap(
            step,
            y,
            amount,
            key,
            incumbent,
            candidate,
            alerts_per_10k=1000,
            block_steps=20,
            n_bootstrap=200,
        )
    except ValueError as exc:
        assert "block_steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
