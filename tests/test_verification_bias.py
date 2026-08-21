from fraud_decisioning.simulate import simulate_payments
from fraud_decisioning.features import build_features
from fraud_decisioning.verification_bias import verification_bias_sensitivity


def test_verification_bias_runs_small():
    df, features = build_features(simulate_payments(n=6000, seed=9, days=60))
    network = {"sender_unique_recipients_24h","recipient_unique_senders_24h","device_unique_senders_24h","pair_tx_24h"}
    raw, summary = verification_bias_sensitivity(
        df, [c for c in features if c not in network], audit_rates=(0.0, 1.0), seeds=(3,)
    )
    assert len(raw) == 2
    assert set(summary.audit_rate) == {0.0, 1.0}
