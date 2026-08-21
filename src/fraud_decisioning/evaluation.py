from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score


def threshold_at_fpr(y, p, target_fpr=0.01):
    legit = np.asarray(p)[np.asarray(y) == 0]
    return float(np.quantile(legit, 1 - target_fpr))


def model_metrics(name, y, p, amount, threshold):
    pred = np.asarray(p) >= threshold
    y = np.asarray(y)
    amount = np.asarray(amount)
    fraud_value = amount[y == 1].sum()
    captured = amount[(y == 1) & pred].sum()
    legit = y == 0
    return {
        "model": name,
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold_at_1pct_fpr": threshold,
        "precision_at_1pct_fpr": precision_score(y, pred, zero_division=0),
        "recall_at_1pct_fpr": recall_score(y, pred, zero_division=0),
        "fraud_value_recall_at_1pct_fpr": captured / fraud_value if fraud_value else np.nan,
        "legit_flag_rate": pred[legit].mean(),
    }


def policy_grid(
    y, p, amount,
    block_efficacy=0.95, review_efficacy=0.65,
    review_case_cost=0.75, false_block_cost_unit=5.0, false_review_cost_unit=1.5,
):
    y = np.asarray(y); p = np.asarray(p); amount = np.asarray(amount)
    rows = []
    for review_t in np.arange(0.03, 0.46, 0.03):
        for block_t in np.arange(max(0.25, review_t + 0.10), 0.96, 0.05):
            block = p >= block_t
            review = (p >= review_t) & ~block
            approve = ~(block | review)
            fraud_value = amount * y
            prevented = (fraud_value[block].sum() * block_efficacy + fraud_value[review].sum() * review_efficacy)
            residual = fraud_value.sum() - prevented
            legit = y == 0
            friction_rate = ((block | review) & legit).sum() / max(1, legit.sum())
            review_cost = review.sum() * review_case_cost
            false_block_cost = (block & legit).sum() * false_block_cost_unit
            false_review_cost = (review & legit).sum() * false_review_cost_unit
            total_cost = residual + review_cost + false_block_cost + false_review_cost
            rows.append({
                "review_threshold": review_t,
                "block_threshold": block_t,
                "fraud_value_prevented": prevented,
                "fraud_value_prevented_rate": prevented / max(1.0, fraud_value.sum()),
                "legit_friction_rate": friction_rate,
                "review_rate": review.mean(),
                "block_rate": block.mean(),
                "residual_fraud_loss": residual,
                "operational_and_friction_cost": review_cost + false_block_cost + false_review_cost,
                "total_policy_cost": total_cost,
            })
    return pd.DataFrame(rows)
