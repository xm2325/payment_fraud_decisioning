from fraud_decisioning.base_rate import precision_from_rates

def test_precision_increases_with_prevalence():
    assert precision_from_rates(0.8, 0.01, 0.01) > precision_from_rates(0.8, 0.01, 0.001)
