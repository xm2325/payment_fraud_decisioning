from fraud_decisioning.simulate import simulate_payments

def test_novel_pattern_is_future_only():
    df = simulate_payments(n=12000, seed=7, days=60)
    novel = df.fraud_type.eq("novel_shared_device_microburst")
    assert novel.any()
    assert (df.loc[novel, "day"] >= 48).all()
    assert not df.loc[df.day < 48, "fraud_type"].eq("novel_shared_device_microburst").any()
