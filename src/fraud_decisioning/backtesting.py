from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from .modeling import fit_lightgbm, fit_sigmoid_calibrator, calibrate
from .evaluation import threshold_at_fpr


def rolling_backtest(df, features, folds=((24,30,36),(30,36,42),(36,42,48),(42,48,54),(48,54,60))):
    """Expanding-window rolling-origin backtest with six-day validation/test blocks."""
    rows = []
    for train_end, val_end, test_end in folds:
        train = df[df.day < train_end]
        val = df[(df.day >= train_end) & (df.day < val_end)]
        test = df[(df.day >= val_end) & (df.day < test_end)]
        if train.is_fraud.nunique() < 2 or val.is_fraud.nunique() < 2 or test.is_fraud.nunique() < 2:
            continue
        model = fit_lightgbm(train[features], train.is_fraud, n_estimators=160)
        pv_raw = model.predict_proba(val[features])[:, 1]
        cal = fit_sigmoid_calibrator(pv_raw, val.is_fraud)
        pv = calibrate(cal, pv_raw)
        pt = calibrate(cal, model.predict_proba(test[features])[:, 1])
        th = threshold_at_fpr(val.is_fraud, pv, 0.01)
        flag = pt >= th
        y = test.is_fraud.to_numpy(int)
        legit = y == 0
        novel = test.fraud_type.eq("novel_shared_device_microburst").to_numpy()
        known = (y == 1) & ~novel
        fv = test.amount.to_numpy(float) * y
        rows.append({
            "train_end_day": train_end,
            "validation_days": f"{train_end}-{val_end-1}",
            "test_days": f"{val_end}-{test_end-1}",
            "test_n": len(test),
            "test_fraud_rate": y.mean(),
            "pr_auc": average_precision_score(y, pt),
            "brier": brier_score_loss(y, pt),
            "threshold": th,
            "legit_flag_rate": flag[legit].mean(),
            "fraud_recall": flag[y == 1].mean(),
            "known_fraud_recall": flag[known].mean() if known.any() else np.nan,
            "novel_fraud_recall": flag[novel].mean() if novel.any() else np.nan,
            "fraud_value_recall": fv[flag].sum() / fv.sum() if fv.sum() else np.nan,
        })
    return pd.DataFrame(rows)


def delayed_label_retraining(df, features, score_start_day=54, label_delay_days=7):
    """Quantify the difference between realistic matured labels and an oracle with instant labels.

    This is a diagnostic for label-latency leakage, not a production estimate.
    """
    test = df[df.day >= score_start_day].copy()
    matured_cutoff = score_start_day - label_delay_days
    delayed_train = df[df.day < matured_cutoff].copy()
    oracle_train = df[df.day < score_start_day].copy()
    # Use a common historical threshold/calibration block that predates the emerging attack.
    common_val = df[(df.day >= 42) & (df.day < 48)].copy()

    rows = []
    for name, train in [(f"asof_{label_delay_days}d_delay", delayed_train), ("oracle_instant_labels", oracle_train)]:
        model = fit_lightgbm(train[features], train.is_fraud, n_estimators=160)
        pv_raw = model.predict_proba(common_val[features])[:, 1]
        cal = fit_sigmoid_calibrator(pv_raw, common_val.is_fraud)
        pv = calibrate(cal, pv_raw)
        pt = calibrate(cal, model.predict_proba(test[features])[:, 1])
        th = threshold_at_fpr(common_val.is_fraud, pv, 0.01)
        flag = pt >= th
        y = test.is_fraud.to_numpy(int)
        novel = test.fraud_type.eq("novel_shared_device_microburst").to_numpy()
        legit = y == 0
        rows.append({
            "training_view": name,
            "available_training_end_day": int(train.day.max()),
            "test_start_day": score_start_day,
            "pr_auc": average_precision_score(y, pt),
            "brier": brier_score_loss(y, pt),
            "legit_flag_rate": flag[legit].mean(),
            "fraud_recall": flag[y == 1].mean(),
            "novel_fraud_recall": flag[novel].mean() if novel.any() else np.nan,
        })
    return pd.DataFrame(rows)
