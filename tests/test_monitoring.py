from fraud_decisioning.monitoring import psi

def test_psi_zero_for_same_distribution():
    x = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]*20
    assert abs(psi(x, x)) < 1e-12
