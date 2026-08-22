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
from fraud_decisioning.paysim_nested_validation import calibration_metrics, nested_policy_cut
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


def _tag(frame: pd.DataFrame, validation_contract: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "validation_contract", validation_contract)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_nested_validation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_nested_features.parquet"
    db = out_dir / "paysim_nested.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
    validate_canonical(audit)
    split = determine_split(con, args.parquet_glob)
    policy_cut = nested_policy_cut(split.train_cut, split.validation_cut)
    materialise_features(con, args.parquet_glob, work)

    train = _load_split(con, work, f"step < {split.train_cut}")
    calibration = _load_split(
        con, work, f"step >= {split.train_cut} AND step < {policy_cut}"
    )
    policy = _load_split(
        con, work, f"step >= {policy_cut} AND step < {split.validation_cut}"
    )
    full_validation = _load_split(
        con, work, f"step >= {split.train_cut} AND step < {split.validation_cut}"
    )
    future = _load_split(con, work, f"step >= {split.validation_cut}")

    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)

    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    policy_raw = model.predict_proba(policy[features])[:, 1]
    full_validation_raw = model.predict_proba(full_validation[features])[:, 1]
    future_raw = model.predict_proba(future[features])[:, 1]

    # Nested contract: calibration labels end before routing-policy labels begin.
    nested_calibrator = fit_sigmoid_calibrator(calibration_raw, calibration.is_fraud)
    calibration_probability = calibrate(nested_calibrator, calibration_raw)
    policy_probability = calibrate(nested_calibrator, policy_raw)
    nested_future_probability = calibrate(nested_calibrator, future_raw)

    # Shared-validation reference reproduces the previous contract for comparison only.
    shared_calibrator = fit_sigmoid_calibrator(full_validation_raw, full_validation.is_fraud)
    shared_validation_probability = calibrate(shared_calibrator, full_validation_raw)
    shared_future_probability = calibrate(shared_calibrator, future_raw)

    diagnostics_rows = []
    for contract, period, frame, probability in (
        ("nested", "calibration_fit", calibration, calibration_probability),
        ("nested", "policy_selection", policy, policy_probability),
        ("nested", "future_test", future, nested_future_probability),
        ("shared_validation", "validation_fit_and_selection", full_validation, shared_validation_probability),
        ("shared_validation", "future_test", future, shared_future_probability),
    ):
        row = calibration_metrics(frame.is_fraud, probability)
        row.update({"validation_contract": contract, "period": period,
                    "step_min": int(frame.step.min()), "step_max": int(frame.step.max())})
        diagnostics_rows.append(row)
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics.to_csv(out_dir / "calibration_diagnostics.csv", index=False)

    reference_capacity = 50
    nested_window_grid = validation_window_alpha_grid(
        policy.step,
        policy.is_fraud,
        policy_probability,
        policy.amount,
        policy.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
        alphas=DEFAULT_ALPHA_GRID,
    )
    nested_window_grid.to_csv(out_dir / "nested_policy_window_alpha_grid.csv", index=False)
    nested_robustness = robustness_summary(nested_window_grid)
    nested_robustness.to_csv(out_dir / "nested_policy_robustness_summary.csv", index=False)
    nested_selected = select_robust_profiles(nested_robustness)
    nested_selected.to_csv(out_dir / "nested_selected_profiles.csv", index=False)

    shared_window_grid = validation_window_alpha_grid(
        full_validation.step,
        full_validation.is_fraud,
        shared_validation_probability,
        full_validation.amount,
        full_validation.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
        alphas=DEFAULT_ALPHA_GRID,
    )
    shared_robustness = robustness_summary(shared_window_grid)
    shared_selected = select_robust_profiles(shared_robustness)
    shared_selected.to_csv(out_dir / "shared_validation_selected_profiles.csv", index=False)

    nested_future = evaluate_selected_profiles(
        nested_selected,
        future.is_fraud,
        nested_future_probability,
        future.amount,
        future.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    shared_future = evaluate_selected_profiles(
        shared_selected,
        future.is_fraud,
        shared_future_probability,
        future.amount,
        future.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    future_comparison = pd.concat(
        [_tag(nested_future, "nested"), _tag(shared_future, "shared_validation")],
        ignore_index=True,
    )
    future_comparison.to_csv(out_dir / "future_contract_comparison.csv", index=False)

    nested_windows = evaluate_profile_windows(
        nested_selected,
        future.step,
        future.is_fraud,
        nested_future_probability,
        future.amount,
        future.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    shared_windows = evaluate_profile_windows(
        shared_selected,
        future.step,
        future.is_fraud,
        shared_future_probability,
        future.amount,
        future.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    pd.concat(
        [_tag(nested_windows, "nested"), _tag(shared_windows, "shared_validation")],
        ignore_index=True,
    ).to_csv(out_dir / "future_contract_windows_50_per_10k.csv", index=False)

    def selection_records(frame: pd.DataFrame, contract: str) -> list[dict]:
        return [
            {
                "validation_contract": contract,
                "profile": str(row.profile),
                "alpha": float(row.alpha),
                "amount_scale": float(row.amount_scale),
            }
            for _, row in frame.iterrows()
        ]

    selection = selection_records(nested_selected, "nested") + selection_records(
        shared_selected, "shared_validation"
    )
    pd.DataFrame(selection).to_csv(out_dir / "selection_comparison.csv", index=False)

    reference_rows = future_comparison.loc[
        future_comparison.target_alerts_per_10k == reference_capacity
    ]
    future_reference = {
        f"{row.validation_contract}:{row.profile}": {
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
        "split": {
            "train_end_step": split.train_cut - 1,
            "calibration_start_step": split.train_cut,
            "calibration_end_step": policy_cut - 1,
            "policy_selection_start_step": policy_cut,
            "policy_selection_end_step": split.validation_cut - 1,
            "future_start_step": split.validation_cut,
        },
        "model": "transaction_plus_relational",
        "selection_capacity_alerts_per_10k": reference_capacity,
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "selection_comparison": selection,
        "future_reference_50_per_10k": future_reference,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The nested contract separates calibration labels from routing-policy selection labels before future evaluation.",
            "The calibration/policy split is the pre-specified midpoint of the original validation period; it is not chosen using future results.",
            "Nested routing selection uses worst-window performance across three contiguous policy-selection windows.",
            "The shared-validation contract is reproduced only as a methodological reference and is not allowed to influence nested policy selection.",
            "Future labels are used only for final evaluation and diagnostics, never calibration or alpha selection.",
            "PaySim is synthetic mobile-money data; results are benchmark evidence rather than production impact or prevented loss."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
