from fraud_decisioning.experiment import two_proportion_sample_size

def test_sample_size_increases_for_smaller_effect():
    n_big_effect = two_proportion_sample_size(0.06, 0.04)
    n_small_effect = two_proportion_sample_size(0.06, 0.05)
    assert n_small_effect > n_big_effect > 0
