from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

def threshold_at_legit_rate(y: np.ndarray, p: np.ndarray, target: float) -> float:
    legit = np.asarray(p)[np.asarray(y) == 0]
    if len(legit) == 0:
        raise ValueError("No legitimate validation rows")
    return float(np.quantile(legit, 1 - target, method="higher"))


def binary_metrics(y, p, amount, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    amount = np.asarray(amount, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("binary_metrics expects finite probabilities in [0, 1]; use rule_metrics for rule/ranking scores")
    pred = p >= threshold
    fraud = y == 1
    legit = ~fraud
    tp = int((pred & fraud).sum())
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(y[pred].mean()) if pred.any() else math.nan,
        "recall": float(tp / fraud.sum()),
        "legit_flag_rate": float(pred[legit].mean()),
        "fraud_value_recall": float(amount[pred & fraud].sum() / amount[fraud].sum()),
        "alerts": int(pred.sum()),
    }


def rule_metrics(y, rule, amount) -> dict:
    y = np.asarray(y, dtype=int)
    rule = np.asarray(rule, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    fraud = y == 1
    legit = ~fraud
    return {
        "precision": float(y[rule].mean()) if rule.any() else math.nan,
        "recall": float(rule[fraud].mean()),
        "legit_flag_rate": float(rule[legit].mean()),
        "fraud_value_recall": float(amount[rule & fraud].sum() / amount[fraud].sum()),
        "alerts": int(rule.sum()),
    }


