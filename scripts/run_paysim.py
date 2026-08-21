from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_decisioning.datasets import load_paysim, paysim_data_audit, canonical_paysim_status
from fraud_decisioning.features import build_features
from fraud_decisioning.modeling import fit_lightgbm, fit_sigmoid_calibrator, calibrate
from fraud_decisioning.evaluation import threshold_at_fpr, model_metrics


def temporal_step_split(df: pd.DataFrame):
    steps = np.sort(df["source_step"].unique())
    if len(steps) < 10:
        raise ValueError("Need at least 10 distinct PaySim time steps for temporal validation")
    train_cut = steps[int(len(steps) * 0.60)]
    val_cut = steps[int(len(steps) * 0.80)]
    train = df[df.source_step < train_cut].copy()
    val = df[(df.source_step >= train_cut) & (df.source_step < val_cut)].copy()
    test = df[df.source_step >= val_cut].copy()
    return train, val, test


def main():
    ap = argparse.ArgumentParser(description="Temporal PaySim external benchmark")
    ap.add_argument("--csv", type=Path, required=True, help="Path to PaySim CSV")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--max-step", type=int, default=None, help="Resource-safe contiguous time-prefix; e.g. 240")
    mode.add_argument("--full", action="store_true", help="Explicitly request all 6.36M rows; full-scale runtime is not validated in this snapshot")
    ap.add_argument("--target-legit-flag-rate", type=float, default=0.001, help="Validation operating point; default 0.1%")
    args = ap.parse_args()

    df = load_paysim(args.csv, max_step=args.max_step)
    audit = paysim_data_audit(df)
    canonical = canonical_paysim_status(df)
    audit["canonical_status"] = canonical
    audit["execution_mode"] = "full" if args.full else f"time_prefix_step_le_{args.max_step}"
    print("PaySim audit:", json.dumps(audit, indent=2))
    if args.full and not canonical["is_canonical_full"]:
        raise ValueError("--full was requested but row/fraud/step counts do not match the standard PaySim dataset")

    df, features = build_features(df)
    paysim_features = [c for c in features if c not in {
        "account_age_days", "device_change", "country_mismatch", "recipient_new",
        "device_activity_24h", "device_unique_senders_24h",
    }]
    train, val, test = temporal_step_split(df)
    if min(train.is_fraud.sum(), val.is_fraud.sum(), test.is_fraud.sum()) == 0:
        raise ValueError("One temporal split has no fraud labels; use a longer --max-step prefix or the full dataset")

    model = fit_lightgbm(train[paysim_features], train.is_fraud)
    pv_raw = model.predict_proba(val[paysim_features])[:, 1]
    cal = fit_sigmoid_calibrator(pv_raw, val.is_fraud)
    pv = calibrate(cal, pv_raw)
    pt = calibrate(cal, model.predict_proba(test[paysim_features])[:, 1])
    threshold = threshold_at_fpr(val.is_fraud, pv, args.target_legit_flag_rate)
    metrics = model_metrics("paysim_lightgbm", test.is_fraud, pt, test.amount, threshold)

    rule = test["is_flagged_fraud"].to_numpy(bool)
    y = test.is_fraud.to_numpy(int)
    legit = y == 0
    fraud = y == 1
    rule_metrics = {
        "model": "paysim_isFlaggedFraud_rule",
        "precision": float(y[rule].mean()) if rule.any() else np.nan,
        "recall": float(rule[fraud].mean()) if fraud.any() else np.nan,
        "legit_flag_rate": float(rule[legit].mean()) if legit.any() else np.nan,
        "fraud_value_recall": float(test.amount.to_numpy(float)[fraud & rule].sum() / test.amount.to_numpy(float)[fraud].sum()),
    }

    out = ROOT / "outputs" / "paysim"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "model_metrics.csv", index=False)
    pd.DataFrame([rule_metrics]).to_csv(out / "source_rule_metrics.csv", index=False)
    pd.DataFrame([
        {"split":"train","n":len(train),"fraud_rate":train.is_fraud.mean(),"step_min":train.source_step.min(),"step_max":train.source_step.max()},
        {"split":"validation","n":len(val),"fraud_rate":val.is_fraud.mean(),"step_min":val.source_step.min(),"step_max":val.source_step.max()},
        {"split":"test","n":len(test),"fraud_rate":test.is_fraud.mean(),"step_min":test.source_step.min(),"step_max":test.source_step.max()},
    ]).to_csv(out / "split_summary.csv", index=False)
    with open(out / "audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({"model": metrics, "source_rule": rule_metrics}, indent=2))


if __name__ == "__main__":
    main()
