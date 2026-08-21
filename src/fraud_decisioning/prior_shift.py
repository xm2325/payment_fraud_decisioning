from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def adjust_probability_for_prior(prob, source_prevalence: float, target_prevalence: float):
    """Adjust calibrated probabilities for a changed class prior.

    This is the standard prior-probability / label-shift correction. It assumes
    P(X|Y) is stable while P(Y) changes. That assumption is testable only with
    deployment data, so this function is a sensitivity tool, not a guarantee.
    """
    p = np.clip(np.asarray(prob, dtype=float), 1e-8, 1 - 1e-8)
    ps = float(np.clip(source_prevalence, 1e-8, 1 - 1e-8))
    pt = float(np.clip(target_prevalence, 1e-8, 1 - 1e-8))
    odds = p / (1 - p)
    prior_ratio = (pt / (1 - pt)) / (ps / (1 - ps))
    new_odds = odds * prior_ratio
    return new_odds / (1 + new_odds)


def _ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include p==1 in the final bin.
    idx = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def prior_shift_sensitivity(y, prob, source_prevalence: float,
                            target_prevalences=(0.001, 0.0025, 0.005, 0.01, 0.02),
                            repeats=20, seed=42):
    """Down-sample one class to emulate deployment prevalences.

    The score distribution is held fixed within each class. Results therefore
    isolate prior/base-rate shift rather than covariate or concept drift.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(prob, dtype=float)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    if not len(i0) or not len(i1):
        raise ValueError("both classes are required")
    rng = np.random.default_rng(seed)
    rows = []
    for target in target_prevalences:
        target = float(target)
        for rep in range(int(repeats)):
            # Use the largest feasible stratified subset at the target prior.
            n1_if_all0 = max(1, int(round(len(i0) * target / (1 - target))))
            if n1_if_all0 <= len(i1):
                s0 = i0
                s1 = rng.choice(i1, size=n1_if_all0, replace=False)
            else:
                n0 = max(1, int(round(len(i1) * (1 - target) / target)))
                s1 = i1
                s0 = rng.choice(i0, size=min(n0, len(i0)), replace=False)
            idx = np.concatenate([s0, s1])
            rng.shuffle(idx)
            ys = y[idx]
            pu = p[idx]
            pa = adjust_probability_for_prior(pu, source_prevalence, target)
            for method, pp in [("unadjusted", pu), ("prior_adjusted", pa)]:
                rows.append({
                    "target_prevalence": target,
                    "repeat": rep,
                    "method": method,
                    "n": int(len(idx)),
                    "observed_prevalence": float(ys.mean()),
                    "mean_predicted_risk": float(pp.mean()),
                    "brier": float(brier_score_loss(ys, pp)),
                    "log_loss": float(log_loss(ys, pp, labels=[0, 1])),
                    "ece_10bin": _ece(ys, pp, 10),
                })
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["target_prevalence", "method"], as_index=False).agg(
        n_mean=("n", "mean"),
        observed_prevalence_mean=("observed_prevalence", "mean"),
        mean_predicted_risk_mean=("mean_predicted_risk", "mean"),
        brier_mean=("brier", "mean"),
        brier_sd=("brier", "std"),
        log_loss_mean=("log_loss", "mean"),
        ece_10bin_mean=("ece_10bin", "mean"),
    )
    return raw, summary
