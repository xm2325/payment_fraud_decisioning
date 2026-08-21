import numpy as np
from fraud_decisioning.prior_shift import adjust_probability_for_prior


def test_prior_adjustment_identity_and_direction():
    p = np.array([0.01, 0.2, 0.8])
    same = adjust_probability_for_prior(p, 0.02, 0.02)
    assert np.allclose(same, p)
    lower = adjust_probability_for_prior(p, 0.02, 0.002)
    higher = adjust_probability_for_prior(p, 0.02, 0.10)
    assert np.all(lower < p)
    assert np.all(higher > p)
