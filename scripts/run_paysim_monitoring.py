from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from fraud_decisioning.modeling import calibrate, fit_lightgbm, fit_sigmoid_calibrator
from fraud_decisioning.paysim_features import FEATURE_SETS
from fraud_decisioning.paysim_full import (
    _load_split,
    audit_sql,
    connect_duckdb,
    determine_split,
    materialise_features,
    validate_canonical,
)
from fraud_decisioning.paysim_metrics import threshold_at_legit_rate
from fraud_decisioning.paysim_monitoring import (
    future_budget_windows,
    locked_threshold_drift,
    posthoc_budget_match,
    recipient_signal_audit,
)


def _json_safe(value):
    """Convert NumPy/Pandas scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_monitoring")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_monitoring_features.parquet"
    db = out_dir / "paysim_monitoring.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
    validate_canonical(audit)
    split = determine_split(con, args.parquet_glob)
    materialise_features(con, args.parquet_glob, work)

    train = _load_split(con, work, f"step < {split.train_cut}")
    val = _load_split(con, work, f"step >= {split.train_cut} AND step < {split.validation_cut}")
    test = _load_split(con, work, f"step >= {split.validation_cut}")

    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
    val_raw = model.predict_proba(val[features])[:, 1]
    calibrator = fit_sigmoid_calibrator(val_raw, val.is_fraud)
    val_score = calibrate(calibrator, val_raw)
    future_score = calibrate(calibrator, model.predict_proba(test[features])[:, 1])

    target = 0.001
    locked_threshold = threshold_at_legit_rate(val.is_fraud.to_numpy(), val_score, target)
    drift = locked_threshold_drift(
        val.is_fraud, val_score, val.amount,
        test.is_fraud, future_score, test.amount,
        locked_threshold, target,
    )
    drift.to_csv(out_dir / "threshold_budget_drift.csv", index=False)

    windows = future_budget_windows(
        test.step, test.is_fraud, future_score, test.amount,
        locked_threshold, target, n_windows=3,
    )
    windows.to_csv(out_dir / "future_budget_windows.csv", index=False)

    oracle = posthoc_budget_match(test.is_fraud, future_score, test.amount, target)
    pd.DataFrame([oracle]).to_csv(out_dir / "future_budget_rethreshold_oracle.csv", index=False)

    recipient = recipient_signal_audit(val, test, target)
    recipient.to_csv(out_dir / "recipient_signal_audit.csv", index=False)

    val_row = drift.loc[drift.period == "validation"].iloc[0]
    future_row = drift.loc[drift.period == "future_test"].iloc[0]
    strongest = recipient.sort_values(["fraud_value_recall", "recall"], ascending=False).iloc[0]
    summary = {
        "audit": audit,
        "split": {"train_cut": split.train_cut, "validation_cut": split.validation_cut},
        "model": "transaction_plus_relational",
        "target_validation_legit_flag_rate": target,
        "locked_threshold": float(locked_threshold),
        "validation_actual_legit_flag_rate": float(val_row.legit_flag_rate),
        "future_actual_legit_flag_rate": float(future_row.legit_flag_rate),
        "future_budget_multiplier_vs_target": float(future_row.budget_multiplier_vs_target),
        "future_budget_multiplier_vs_validation_actual": float(future_row.budget_multiplier_vs_validation_actual),
        "locked_threshold_future_precision": float(future_row.precision),
        "locked_threshold_future_recall": float(future_row.recall),
        "locked_threshold_future_fraud_value_recall": float(future_row.fraud_value_recall),
        "posthoc_budget_match": oracle,
        "strongest_recipient_signal_by_future_value_recall": strongest.to_dict(),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The operating threshold is selected on validation labels only and then locked for future evaluation.",
            "The post-hoc future budget-matched threshold is diagnostic-only and is not a deployable prospective result.",
            "Recipient signals use strict prior-step history; same-step PaySim transactions are simultaneous.",
            "PaySim has no confirmed mule-account label. Recipient activity results are investigation-signal audits, not mule-classification claims.",
            "PaySim is synthetic mobile-money data; all reported performance is benchmark evidence, not production impact."
        ]
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
