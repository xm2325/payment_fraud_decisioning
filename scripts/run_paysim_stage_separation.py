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
from fraud_decisioning.paysim_routing_profiles import (
    DEFAULT_ALPHA_GRID,
    evaluate_profile_windows,
    evaluate_selected_profiles,
)
from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)
from fraud_decisioning.paysim_stage_separation import (
    probability_stage_metrics,
    split_summary_frame,
    split_validation_stages,
    stage_masks,
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
    parser.add_argument("--out", default="results/paysim_stage_separation")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_stage_separation_features.parquet"
    db = out_dir / "paysim_stage_separation.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
    validate_canonical(audit)
    split = determine_split(con, args.parquet_glob)
    materialise_features(con, args.parquet_glob, work)

    train = _load_split(con, work, f"step < {split.train_cut}")
    validation = _load_split(
        con, work, f"step >= {split.train_cut} AND step < {split.validation_cut}"
    )
    future = _load_split(con, work, f"step >= {split.validation_cut}")

    stage_split = split_validation_stages(
        validation.step, calibration_fraction=args.calibration_fraction
    )
    calibration_mask, policy_mask = stage_masks(validation.step, stage_split)
    calibration = validation.loc[calibration_mask].copy()
    policy = validation.loc[policy_mask].copy()
    stage_summary = split_summary_frame(validation.step, validation.is_fraud, stage_split)
    stage_summary.to_csv(out_dir / "validation_stage_split.csv", index=False)

    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = fit_sigmoid_calibrator(calibration_raw, calibration.is_fraud)
    calibration_probability = calibrate(calibrator, calibration_raw)
    policy_probability = calibrate(
        calibrator, model.predict_proba(policy[features])[:, 1]
    )
    future_probability = calibrate(
        calibrator, model.predict_proba(future[features])[:, 1]
    )

    probability_rows = [
        probability_stage_metrics("calibration", calibration.is_fraud, calibration_probability),
        probability_stage_metrics("policy_selection", policy.is_fraud, policy_probability),
        probability_stage_metrics("future_test", future.is_fraud, future_probability),
    ]
    probability_frame = pd.DataFrame(probability_rows)
    probability_frame.to_csv(out_dir / "probability_stage_diagnostics.csv", index=False)

    reference_capacity = 50
    window_grid = validation_window_alpha_grid(
        policy.step,
        policy.is_fraud,
        policy_probability,
        policy.amount,
        policy.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
        alphas=DEFAULT_ALPHA_GRID,
    )
    window_grid.to_csv(out_dir / "policy_window_alpha_grid.csv", index=False)
    policy_robustness = robustness_summary(window_grid)
    policy_robustness.to_csv(out_dir / "policy_robustness_summary.csv", index=False)
    selected = select_robust_profiles(policy_robustness)
    selected["selection_split"] = "policy_selection_windows_only"
    selected.to_csv(out_dir / "selected_profiles.csv", index=False)

    future_frontier = evaluate_selected_profiles(
        selected,
        future.is_fraud,
        future_probability,
        future.amount,
        future.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    future_frontier.to_csv(out_dir / "future_frontier.csv", index=False)
    future_windows = evaluate_profile_windows(
        selected,
        future.step,
        future.is_fraud,
        future_probability,
        future.amount,
        future.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    future_windows.to_csv(out_dir / "future_windows_50_per_10k.csv", index=False)

    selected_rows = [
        {
            "profile": str(row.profile),
            "alpha": float(row.alpha),
            "amount_scale": float(row.amount_scale),
            "min_recall": float(row.min_recall),
            "min_fraud_value_recall": float(row.min_fraud_value_recall),
            "min_balanced_hmean": float(row.min_balanced_hmean),
        }
        for _, row in selected.iterrows()
    ]
    reference_rows = future_frontier.loc[
        future_frontier.target_alerts_per_10k == reference_capacity
    ]
    future_reference = {
        str(row.profile): {
            "alpha": float(row.alpha),
            "precision": float(row.precision),
            "fraud_recall": float(row.recall),
            "fraud_value_recall": float(row.fraud_value_recall),
            "legitimate_alerts_per_10k": float(row.legit_alerts_per_10k),
        }
        for _, row in reference_rows.iterrows()
    }

    summary = {
        "audit": audit,
        "outer_split": {"train_cut": split.train_cut, "future_cut": split.validation_cut},
        "validation_stage_split": {
            "calibration_fraction": float(args.calibration_fraction),
            "calibration_step_min": stage_split.calibration_step_min,
            "calibration_step_max": stage_split.calibration_step_max,
            "policy_step_min": stage_split.policy_step_min,
            "policy_step_max": stage_split.policy_step_max,
            "policy_cut": stage_split.policy_cut,
        },
        "model": "transaction_plus_relational_fixed_before_stage_separation",
        "selection_capacity_alerts_per_10k": reference_capacity,
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "policy_selection_windows": 3,
        "selected_profiles": selected_rows,
        "future_reference_50_per_10k": future_reference,
        "probability_stage_diagnostics": probability_rows,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The predictive feature family is fixed before this experiment; v1.7 tests calibration-versus-routing separation, not feature-family selection.",
            "Only steps 446-519 fit the probability calibrator under the canonical 50/50 validation-stage split.",
            "Only later validation steps 520-594 select routing alpha and its worst-window robustness objective.",
            "Future steps 595-743 evaluate the frozen calibrator and routing policy; future labels never select alpha.",
            "The amount scale is fitted on policy-selection amounts only and frozen for future routing.",
            "PaySim is synthetic mobile-money data; results are benchmark evidence rather than production impact, prevented loss or staffing estimates."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
