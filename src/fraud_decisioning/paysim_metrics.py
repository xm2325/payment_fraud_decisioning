from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def threshold_at_legit_rate(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Choose the lowest >= threshold whose validation legitimate-alert rate stays <= target.

    Tree/rule scores can contain large ties. A plain quantile combined with ``score >= threshold``
    can therefore exceed the requested alert budget. This selector treats the target as a hard cap:
    if the boundary score is tied across too many legitimate rows, it moves just above that score.
    """
    if not 0 <= target <= 1:
        raise ValueError("target must be in [0, 1]")
    scores = np.asarray(p, dtype=float)
    labels = np.asarray(y)
    legit = scores[labels == 0]
    if len(legit) == 0:
        raise ValueError("No legitimate validation rows")
    if np.any(~np.isfinite(legit)):
        raise ValueError("Legitimate validation scores must be finite")
    if target == 1:
        return float(np.min(legit))

    max_alerts = int(np.floor(target * len(legit)))
    if max_alerts <= 0:
        return float(np.nextafter(np.max(legit), np.inf))

    descending = np.sort(legit)[::-1]
    boundary = float(descending[max_alerts - 1])
    flagged_at_boundary = int(np.count_nonzero(legit >= boundary))
    if flagged_at_boundary > max_alerts:
        return float(np.nextafter(boundary, np.inf))
    return boundary


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
