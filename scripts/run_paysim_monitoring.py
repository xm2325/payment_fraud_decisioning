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
    posthoc_threshold_cap,
    ranked_capacity_frontier,
    ranked_capacity_windows,
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

    # Scalar-threshold diagnostic: target is a validation legitimate-alert hard cap.
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
    windows.to_csv(out_dir / "future_threshold_windows.csv", index=False)

    threshold_cap = posthoc_threshold_cap(test.is_fraud, future_score, test.amount, target)
    pd.DataFrame([threshold_cap]).to_csv(out_dir / "future_posthoc_threshold_cap.csv", index=False)

    # Deployable capacity contract: exact total alert budget, independent of labels.
    capacity = ranked_capacity_frontier(
        test.is_fraud,
        future_score,
        test.amount,
        test.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    capacity.to_csv(out_dir / "ranked_capacity_frontier.csv", index=False)

    reference_capacity_per_10k = 50
    capacity_windows = ranked_capacity_windows(
        test.step,
        test.is_fraud,
        future_score,
        test.amount,
        test.event_key,
        alerts_per_10k=reference_capacity_per_10k,
        n_windows=3,
    )
    capacity_windows.to_csv(out_dir / "ranked_capacity_windows_50_per_10k.csv", index=False)

    recipient = recipient_signal_audit(val, test, target)
    recipient.to_csv(out_dir / "recipient_signal_audit.csv", index=False)

    val_row = drift.loc[drift.period == "validation"].iloc[0]
    future_row = drift.loc[drift.period == "future_test"].iloc[0]
    capacity_reference = capacity.loc[
        capacity.target_alerts_per_10k == reference_capacity_per_10k
    ].iloc[0]
    strongest = recipient.sort_values(["fraud_value_recall", "recall"], ascending=False).iloc[0]
    summary = {
        "audit": audit,
        "split": {"train_cut": split.train_cut, "validation_cut": split.validation_cut},
        "model": "transaction_plus_relational",
        "scalar_threshold_diagnostic": {
            "target_validation_legit_flag_rate": target,
            "locked_threshold": float(locked_threshold),
            "validation_actual_legit_flag_rate": float(val_row.legit_flag_rate),
            "future_actual_legit_flag_rate": float(future_row.legit_flag_rate),
            "future_precision": float(future_row.precision),
            "future_recall": float(future_row.recall),
            "future_fraud_value_recall": float(future_row.fraud_value_recall),
            "posthoc_threshold_cap": threshold_cap,
        },
        "ranked_capacity_reference": capacity_reference.to_dict(),
        "ranked_capacity_reference_alerts_per_10k": reference_capacity_per_10k,
        "strongest_recipient_signal_by_future_value_recall": strongest.to_dict(),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "Scalar threshold targets are hard caps; tied scores can materially under-use a narrow budget.",
            "Ranked capacity routing fixes a total alert budget per 10,000 transactions without using fraud labels to set capacity.",
            "Equal model scores are resolved only by a stable event key derived from non-label transaction fields; event_key is never a model feature.",
            "The post-hoc future threshold cap is retrospective and diagnostic-only; it is not a deployable prospective result.",
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
