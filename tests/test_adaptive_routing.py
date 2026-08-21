import numpy as np
import pandas as pd

from fraud_decisioning.adaptive_routing import adaptive_capacity_routing


def _fixture(n=24):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="10min"),
        "is_fraud": [1 if i % 5 == 0 else 0 for i in range(n)],
        "amount": np.linspace(10, 100, n),
        "fraud_type": ["novel_shared_device_microburst" if i in (5, 15) else ("known" if i % 5 == 0 else "legit") for i in range(n)],
    })


def test_adaptive_routing_never_exceeds_scaled_capacity():
    df = _fixture()
    p = np.linspace(0.0, 0.9, len(df))
    a = np.linspace(0.9, 0.0, len(df))
    out = adaptive_capacity_routing(
        df, p, a, review_threshold=0.1, block_threshold=0.95,
        anomaly_threshold=0.1, analyst_capacity_per_hour=3,
        traffic_multipliers=(1.0, 2.0), exploration_share=0.2,
    )
    assert (out["capacity_utilisation_after_admission"] <= 1.0 + 1e-12).all()
    assert out.loc[out.traffic_multiplier.eq(2.0), "candidate_acceptance_rate"].iloc[0] <= out.loc[out.traffic_multiplier.eq(1.0), "candidate_acceptance_rate"].iloc[0]


def test_adaptive_routing_validates_inputs():
    df = _fixture(6)
    p = np.ones(len(df)) * 0.2
    a = np.ones(len(df)) * 0.2
    try:
        adaptive_capacity_routing(df, p, a, 0.1, 0.9, 0.1, analyst_capacity_per_hour=0)
        assert False
    except ValueError:
        pass
