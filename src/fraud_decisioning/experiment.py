from __future__ import annotations
from statistics import NormalDist
import math


def two_proportion_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Approximate per-arm sample size for a two-sided difference in proportions."""
    if not (0 < p1 < 1 and 0 < p2 < 1 and p1 != p2):
        raise ValueError("p1 and p2 must be distinct probabilities in (0, 1)")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2*pbar*(1-pbar)) + z_beta * math.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    return math.ceil(num / ((p1 - p2) ** 2))
