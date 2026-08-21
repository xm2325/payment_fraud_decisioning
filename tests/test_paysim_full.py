from fraud_decisioning.paysim_full import CANONICAL_PAYSIM, FEATURE_COLUMNS, feature_sql, validate_canonical


def test_feature_sql_uses_strict_prior_step_windows():
    sql = feature_sql("/tmp/paysim-*.parquet")
    assert "RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING" in sql
    assert "RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING" in sql
    assert "RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING" in sql
    assert "isFlaggedFraud" in sql
    for feature in FEATURE_COLUMNS:
        assert feature in sql


def test_canonical_audit_contract():
    audit = dict(CANONICAL_PAYSIM)
    audit["fraud_rate"] = audit["fraud_n"] / audit["n"]
    validate_canonical(audit)


def test_paysim_feature_ablation_excludes_simulator_balances():
    from fraud_decisioning.paysim_features import FEATURE_SETS

    transaction = FEATURE_SETS["transaction_only"]
    history = FEATURE_SETS["transaction_plus_history"]
    full = FEATURE_SETS["full_with_simulator_balances"]

    for name in ("balance_fraction", "orig_balance_delta", "dest_balance_delta"):
        assert name not in transaction
        assert name not in history
        assert name in full
    assert set(transaction) < set(history) < set(full)


def test_relational_feature_set_is_balance_free_and_strictly_larger():
    from fraud_decisioning.paysim_features import BALANCE_FREE_CANDIDATES, FEATURE_SETS

    history = FEATURE_SETS["transaction_plus_history"]
    relational = FEATURE_SETS["transaction_plus_relational"]
    full = FEATURE_SETS["full_with_simulator_balances"]
    assert BALANCE_FREE_CANDIDATES == ["transaction_only", "transaction_plus_history", "transaction_plus_relational"]
    assert set(history) < set(relational) < set(full)
    for name in ("balance_fraction", "orig_balance_delta", "dest_balance_delta"):
        assert name not in relational
    for name in (
        "pair_tx_7d",
        "pair_amount_7d",
        "sender_recipient_share_7d",
        "sender_unique_recipients_7d",
        "recipient_unique_senders_7d",
    ):
        assert name in relational


def test_relational_sql_uses_prior_step_windows():
    sql = feature_sql("/tmp/paysim-*.parquet")
    assert "PARTITION BY nameOrig, nameDest ORDER BY step" in sql
    assert "APPROX_COUNT_DISTINCT(nameDest) OVER" in sql
    assert "APPROX_COUNT_DISTINCT(nameOrig) OVER" in sql
    assert "RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING" in sql


def test_balance_free_champion_uses_validation_pr_auc_only():
    from fraud_decisioning.paysim_full import select_balance_free_champion

    rows = [
        {"model": "transaction_only", "validation_pr_auc": 0.20, "pr_auc": 0.99},
        {"model": "transaction_plus_history", "validation_pr_auc": 0.25, "pr_auc": 0.10},
        {"model": "transaction_plus_relational", "validation_pr_auc": 0.30, "pr_auc": 0.15},
        {"model": "full_with_simulator_balances", "validation_pr_auc": 0.99, "pr_auc": 0.99},
    ]
    assert select_balance_free_champion(rows) == "transaction_plus_relational"


def test_amount_type_rule_score_is_explicitly_not_probability():
    import pandas as pd
    from fraud_decisioning.paysim_full import amount_type_rule_scores

    df = pd.DataFrame({
        "type_TRANSFER": [1, 0, 0],
        "type_CASH_OUT": [0, 1, 0],
        "log_amount": [12.0, 8.0, 4.0],
    })
    scores = amount_type_rule_scores(df)
    assert scores.tolist() == [12.0, 8.0, -1e9]


def test_fraud_value_concentration_contract():
    import numpy as np
    from fraud_decisioning.paysim_full import fraud_value_concentration

    y = np.array([1] * 10 + [0])
    amount = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 999], dtype=float)
    out = fraud_value_concentration(amount, y)
    assert out["top_10pct_cases_value_share"] == 10 / 55
    assert out["top_50pct_cases_value_share"] == 40 / 55


def test_binary_metrics_rejects_unbounded_rule_scores():
    import numpy as np
    import pytest
    from fraud_decisioning.paysim_metrics import binary_metrics

    with pytest.raises(ValueError, match="probabilities"):
        binary_metrics(np.array([0, 1]), np.array([0.2, 3.0]), np.array([1.0, 1.0]), 0.5)


def test_threshold_at_legit_rate_respects_budget_with_boundary_ties():
    import numpy as np
    from fraud_decisioning.paysim_metrics import threshold_at_legit_rate

    y = np.array([0] * 1000 + [1])
    scores = np.array([0.0] * 998 + [0.9, 0.9] + [0.95])
    threshold = threshold_at_legit_rate(y, scores, 0.001)
    legit_flag_rate = np.mean(scores[y == 0] >= threshold)
    assert legit_flag_rate <= 0.001
    assert legit_flag_rate == 0.0


def test_threshold_at_legit_rate_uses_available_budget_without_tie():
    import numpy as np
    from fraud_decisioning.paysim_metrics import threshold_at_legit_rate

    y = np.array([0] * 1000 + [1])
    legit_scores = np.linspace(0.0, 0.999, 1000)
    scores = np.concatenate([legit_scores, [0.5]])
    threshold = threshold_at_legit_rate(y, scores, 0.001)
    assert np.mean(scores[y == 0] >= threshold) == 0.001
