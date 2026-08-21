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
from fraud_decisioning.paysim_routing_profiles import (
    DEFAULT_ALPHA_GRID,
    alpha_grid_metrics,
    priority_score,
    select_profiles,
)
from fraud_decisioning.paysim_surge_capacity import (
    DEFAULT_TRIGGER_GRID,
    evaluate_surge_windows,
    surge_trigger_sensitivity,
    validation_tail_reference,
)


def _json_safe(value):
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
    parser.add_argument("--out", default="results/paysim_surge_capacity")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_surge_features.parquet"
    db = out_dir / "paysim_surge.duckdb"
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
    val_probability = calibrate(calibrator, val_raw)
    future_probability = calibrate(calibrator, model.predict_proba(test[features])[:, 1])

    validation_grid = alpha_grid_metrics(
        val.is_fraud,
        val_probability,
        val.amount,
        val.event_key,
        alerts_per_10k=50,
        alphas=DEFAULT_ALPHA_GRID,
    )
    selected = select_profiles(validation_grid)
    balanced = selected.loc[selected.profile == "balanced"].iloc[0]
    alpha = float(balanced.alpha)
    amount_scale = float(balanced.amount_scale)
    future_priority = priority_score(future_probability, test.amount, alpha, amount_scale)

    tail_threshold, validation_tail_rate = validation_tail_reference(
        val_probability,
        tail_quantile=0.995,
    )
    windows = evaluate_surge_windows(
        test.step,
        test.is_fraud,
        future_probability,
        future_priority,
        test.amount,
        test.event_key,
        tail_threshold=tail_threshold,
        validation_tail_rate=validation_tail_rate,
        baseline_alerts_per_10k=50,
        surge_alerts_per_10k=100,
        trigger_multiplier=1.5,
        n_windows=3,
    )
    windows.to_csv(out_dir / "surge_capacity_windows.csv", index=False)

    sensitivity = surge_trigger_sensitivity(
        test.step,
        test.is_fraud,
        future_probability,
        future_priority,
        test.amount,
        test.event_key,
        tail_threshold=tail_threshold,
        validation_tail_rate=validation_tail_rate,
        baseline_alerts_per_10k=50,
        surge_alerts_per_10k=100,
        trigger_multipliers=DEFAULT_TRIGGER_GRID,
        n_windows=3,
    )
    sensitivity.to_csv(out_dir / "surge_trigger_sensitivity.csv", index=False)

    comparison = []
    for period in windows.period.unique():
        fixed = windows[(windows.period == period) & (windows.policy == "fixed_baseline")].iloc[0]
        surge = windows[(windows.period == period) & (windows.policy == "score_tail_surge")].iloc[0]
        comparison.append({
            "period": period,
            "surge_triggered": bool(surge.surge_triggered),
            "score_tail_multiplier_vs_validation": float(surge.score_tail_multiplier_vs_validation),
            "fixed_capacity_per_10k": float(fixed.target_alerts_per_10k),
            "surge_capacity_per_10k": float(surge.target_alerts_per_10k),
            "added_review_slots": int(surge.alerts - fixed.alerts),
            "precision_delta": float(surge.precision - fixed.precision),
            "fraud_recall_delta": float(surge.recall - fixed.recall),
            "fraud_value_recall_delta": float(surge.fraud_value_recall - fixed.fraud_value_recall),
        })

    sensitivity_summary = []
    for trigger in DEFAULT_TRIGGER_GRID:
        rows = sensitivity[sensitivity.trigger_multiplier == trigger]
        sensitivity_summary.append({
            "trigger_multiplier": float(trigger),
            "triggered_windows": int(rows.surge_triggered.sum()),
            "added_review_slots": int(rows.added_review_slots.sum()),
            "triggered_periods": rows.loc[rows.surge_triggered, "period"].tolist(),
        })

    with open(out_dir / "surge_summary.json", "w") as f:
        json.dump(_json_safe({
            "audit": audit,
            "model": "transaction_plus_relational",
            "routing_alpha_selected_on_validation": alpha,
            "routing_amount_scale": amount_scale,
            "tail_quantile": 0.995,
            "validation_tail_threshold": tail_threshold,
            "validation_actual_tail_rate": validation_tail_rate,
            "reference_trigger_multiplier": 1.5,
            "trigger_sensitivity_grid": list(DEFAULT_TRIGGER_GRID),
            "baseline_alerts_per_10k": 50,
            "surge_alerts_per_10k": 100,
            "window_comparison_at_reference_trigger": comparison,
            "trigger_sensitivity_summary": sensitivity_summary,
            "runtime_seconds": float(time.time() - started),
            "boundaries": [
                "The surge trigger uses model-score load only; future fraud labels are not used to choose capacity.",
                "The 0.5% tail, 1.5x reference trigger, 1.5/2.0/2.5 sensitivity grid and 50-to-100 capacity step are pre-specified stress-test assumptions, not Moniepoint staffing rules.",
                "Trigger sensitivity is reported as governance sensitivity; no future outcome is used to select an optimal trigger.",
                "The routing alpha is selected on validation before future evaluation.",
                "Increasing review capacity is an operational scenario, not a claim that additional analysts are instantly available.",
                "PaySim is synthetic mobile-money data; reported changes are benchmark evidence, not prevented-loss or production impact."
            ],
        }), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
