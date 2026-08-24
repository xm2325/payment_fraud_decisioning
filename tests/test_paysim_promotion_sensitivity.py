import pytest

from fraud_decisioning.paysim_promotion_sensitivity import (
    classify_dependence_sensitivity,
    sensitivity_summary,
)


def test_classify_dependence_sensitivity():
    assert classify_dependence_sensitivity(["KEEP_INCUMBENT"] * 4) == "ROBUST_KEEP_INCUMBENT"
    assert classify_dependence_sensitivity(["PROMOTE"] * 4) == "ROBUST_PROMOTE"
    assert (
        classify_dependence_sensitivity(
            ["KEEP_INCUMBENT", "KEEP_INCUMBENT", "PROMOTE", "KEEP_INCUMBENT"]
        )
        == "DEPENDENCE_SENSITIVE"
    )


def test_classify_rejects_empty_or_unknown_values():
    with pytest.raises(ValueError):
        classify_dependence_sensitivity([])
    with pytest.raises(ValueError):
        classify_dependence_sensitivity(["NO_POLICY_CHANGE"])


def test_sensitivity_summary_orders_blocks_and_tracks_bounds():
    rows = [
        {
            "cycle": 3,
            "profile": "value_first",
            "block_steps": 10,
            "decision": "KEEP_INCUMBENT",
            "primary_lower_bound": 0.08,
            "lcb_fraud_recall": -0.04,
        },
        {
            "cycle": 3,
            "profile": "value_first",
            "block_steps": 1,
            "decision": "PROMOTE",
            "primary_lower_bound": 0.12,
            "lcb_fraud_recall": -0.01,
        },
        {
            "cycle": 3,
            "profile": "value_first",
            "block_steps": 5,
            "decision": "KEEP_INCUMBENT",
            "primary_lower_bound": 0.10,
            "lcb_fraud_recall": -0.03,
        },
        {
            "cycle": 3,
            "profile": "value_first",
            "block_steps": 3,
            "decision": "PROMOTE",
            "primary_lower_bound": 0.11,
            "lcb_fraud_recall": -0.015,
        },
    ]
    result = sensitivity_summary(rows)
    assert len(result) == 1
    row = result[0]
    assert row["sensitivity_class"] == "DEPENDENCE_SENSITIVE"
    assert row["block_steps"] == "1,3,5,10"
    assert row["min_primary_lower_bound"] == pytest.approx(0.08)
    assert row["max_primary_lower_bound"] == pytest.approx(0.12)
    assert row["min_fraud_recall_lcb"] == pytest.approx(-0.04)
    assert row["max_fraud_recall_lcb"] == pytest.approx(-0.01)
