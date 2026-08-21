from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .modeling import fit_lightgbm, fit_sigmoid_calibrator, calibrate
from .evaluation import threshold_at_fpr


def verification_bias_sensitivity(
    df,
    features,
    audit_rates=(0.0, 0.01, 0.05, 0.10, 1.0),
    seeds=(11, 23),
):
    """Stress-test training-label coverage under investigation-driven verification.

    Historical transactions with device change, country mismatch, or top-3%
    amount are assumed to receive follow-up labels. A governance-set random
    audit rate supplies labels outside that risk-triggered set. The validation
    block is treated as an independently audited holdout for comparable
    calibration and thresholds.

    The mechanism is synthetic and intentionally simple. It demonstrates label
    coverage risk; it is not an estimate of a real fraud team's review process.
    """
    train = df[df.day < 36].copy()
    val = df[(df.day >= 36) & (df.day < 48)].copy()
    test = df[df.day >= 48].copy()
    amount_cut = float(train.amount.quantile(0.97))
    triggered = (
        train.device_change.eq(1)
        | train.country_mismatch.eq(1)
        | train.amount.ge(amount_cut)
    ).to_numpy()
    known_test = ~test.fraud_type.eq("novel_shared_device_microburst").to_numpy()
    known_y = test.is_fraud.to_numpy(int)[known_test]

    rows = []
    for audit in audit_rates:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            observed = triggered | (rng.random(len(train)) < float(audit))
            labelled = train.loc[observed].copy()
            if labelled.is_fraud.nunique() < 2:
                continue
            model = fit_lightgbm(labelled[features], labelled.is_fraud, n_estimators=160)
            pv_raw = model.predict_proba(val[features])[:, 1]
            cal = fit_sigmoid_calibrator(pv_raw, val.is_fraud)
            pv = calibrate(cal, pv_raw)
            pt = calibrate(cal, model.predict_proba(test[features])[:, 1])
            th = threshold_at_fpr(val.is_fraud, pv, 0.01)
            flag = pt >= th
            legit = test.is_fraud.eq(0).to_numpy()
            rec = {}
            for typ in ("account_takeover", "transfer_burst", "mule_cashout"):
                mask = test.fraud_type.eq(typ).to_numpy()
                rec[f"recall_{typ}"] = float(flag[mask].mean()) if mask.any() else np.nan
            rows.append({
                "audit_rate": float(audit),
                "seed": int(seed),
                "labelled_train_n": int(len(labelled)),
                "labelled_train_rate": float(observed.mean()),
                "labelled_fraud_rate": float(labelled.is_fraud.mean()),
                "labelled_account_takeover_n": int(labelled.fraud_type.eq("account_takeover").sum()),
                "labelled_transfer_burst_n": int(labelled.fraud_type.eq("transfer_burst").sum()),
                "labelled_mule_cashout_n": int(labelled.fraud_type.eq("mule_cashout").sum()),
                "known_test_pr_auc": float(average_precision_score(known_y, pt[known_test])),
                "legit_flag_rate": float(flag[legit].mean()),
                **rec,
            })
    raw = pd.DataFrame(rows)
    metric_cols = [
        "labelled_train_n", "labelled_train_rate", "labelled_fraud_rate",
        "labelled_account_takeover_n", "labelled_transfer_burst_n", "labelled_mule_cashout_n",
        "known_test_pr_auc", "legit_flag_rate", "recall_account_takeover",
        "recall_transfer_burst", "recall_mule_cashout",
    ]
    summary = raw.groupby("audit_rate", as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["audit_rate"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    return raw, summary
