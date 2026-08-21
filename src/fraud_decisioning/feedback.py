from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .modeling import fit_lightgbm, fit_sigmoid_calibrator, calibrate
from .evaluation import threshold_at_fpr
from .novelty import fit_tail_detector, tail_score


def analyst_feedback_curve(df, features, review_budgets=(0, 10, 50, 100, 200, 500)):
    """Simulate an expedited analyst-feedback loop for an emerging attack.

    Historical model training ends before day 42; days 42-47 are calibration.
    The anomaly channel ranks cases during days 48-53. Analysts review the top-k
    anomaly cases and their simulator truth is treated as expedited feedback.
    The model is retrained and evaluated only on later days 54-59.

    This is a controlled simulation of feedback value, not a claim about real
    investigator speed, chargeback maturity, or Moniepoint operations.
    """
    base = df[df.day < 42].copy()
    val = df[(df.day >= 42) & (df.day < 48)].copy()
    discovery = df[(df.day >= 48) & (df.day < 54)].copy()
    future = df[df.day >= 54].copy()

    scales = fit_tail_detector(base[base.is_fraud == 0])
    discovery_score = tail_score(scales, discovery)
    order = np.argsort(-discovery_score)

    rows = []
    for budget in review_budgets:
        reviewed_idx = order[: min(int(budget), len(discovery))]
        feedback = discovery.iloc[reviewed_idx].copy()
        train = pd.concat([base, feedback], ignore_index=True)
        model = fit_lightgbm(train[features], train.is_fraud, n_estimators=160)
        pv_raw = model.predict_proba(val[features])[:, 1]
        cal = fit_sigmoid_calibrator(pv_raw, val.is_fraud)
        pv = calibrate(cal, pv_raw)
        pf = calibrate(cal, model.predict_proba(future[features])[:, 1])
        th = threshold_at_fpr(val.is_fraud, pv, 0.01)
        flag = pf >= th
        y = future.is_fraud.to_numpy(int)
        novel = future.fraud_type.eq("novel_shared_device_microburst").to_numpy()
        legit = y == 0
        amt = future.amount.to_numpy(float)
        total_value = amt[y == 1].sum()
        rows.append({
            "analyst_review_budget": int(budget),
            "feedback_fraud_n": int(feedback.is_fraud.sum()) if len(feedback) else 0,
            "feedback_novel_fraud_n": int(feedback.fraud_type.eq("novel_shared_device_microburst").sum()) if len(feedback) else 0,
            "future_pr_auc": float(average_precision_score(y, pf)),
            "future_legit_flag_rate": float(flag[legit].mean()),
            "future_fraud_recall": float(flag[y == 1].mean()),
            "future_novel_fraud_recall": float(flag[novel].mean()) if novel.any() else np.nan,
            "future_fraud_value_recall": float(amt[(y == 1) & flag].sum() / total_value) if total_value else np.nan,
        })
    return pd.DataFrame(rows)
