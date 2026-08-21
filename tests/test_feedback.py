from fraud_decisioning.simulate import simulate_payments
from fraud_decisioning.features import build_features
from fraud_decisioning.feedback import analyst_feedback_curve

def test_feedback_curve_runs_small():
    df, features = build_features(simulate_payments(n=6000, seed=7, days=60))
    network = {"sender_unique_recipients_24h","recipient_unique_senders_24h","device_unique_senders_24h","pair_tx_24h"}
    out = analyst_feedback_curve(df, [c for c in features if c not in network], review_budgets=(0,10))
    assert out.analyst_review_budget.tolist() == [0,10]
    assert out.future_pr_auc.notna().all()
