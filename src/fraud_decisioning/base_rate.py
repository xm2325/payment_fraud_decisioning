from __future__ import annotations
import numpy as np
import pandas as pd


def precision_from_rates(tpr: float, fpr: float, prevalence: float) -> float:
    denom = tpr * prevalence + fpr * (1.0 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else np.nan


def prevalence_sensitivity(y_true, flag, prevalences=(0.001, 0.0025, 0.005, 0.01, 0.02)):
    """Translate measured TPR/FPR to expected precision at alternate base rates.

    This uses Bayes' rule and assumes conditional detector rates remain fixed.
    It is a planning sensitivity analysis, not an external validation result.
    """
    y = np.asarray(y_true, dtype=int)
    f = np.asarray(flag, dtype=bool)
    tpr = float(f[y == 1].mean())
    fpr = float(f[y == 0].mean())
    return pd.DataFrame([
        {"fraud_prevalence": float(p), "measured_tpr": tpr, "measured_fpr": fpr,
         "expected_alert_precision": precision_from_rates(tpr, fpr, float(p))}
        for p in prevalences
    ])
