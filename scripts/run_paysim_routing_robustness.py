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
    alpha_grid_metrics,
    evaluate_profile_windows,
    evaluate_selected_profiles,
    select_profiles,
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


def _tag(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "selection", selection)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_routing_robustness")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_routing_robustness_features.parquet"
    db = out_dir / "paysim_routing_robustness.duckdb"
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

    reference_capacity = 50
    aggregate_grid = alpha_grid_metrics(
        val.is_fraud,
        val_probability,
        val.amount,
        val.event_key,
        alerts_per_10k=reference_capacity,
        alphas=DEFAULT_ALPHA_GRID,
    )
    aggregate_selected = select_profiles(aggregate_grid)
    aggregate_selected.to_csv(out_dir / "aggregate_selected_profiles.csv", index=False)

    window_grid = validation_window_alpha_grid(
        val.step,
        val.is_fraud,
        val_probability,
        val.amount,
        val.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
        alphas=DEFAULT_ALPHA_GRID,
    )
    window_grid.to_csv(out_dir / "validation_window_alpha_grid.csv", index=False)

    robust_summary = robustness_summary(window_grid)
    robust_summary.to_csv(out_dir / "validation_robustness_summary.csv", index=False)
    robust_selected = select_robust_profiles(robust_summary)
    robust_selected.to_csv(out_dir / "robust_selected_profiles.csv", index=False)

    future_aggregate = evaluate_selected_profiles(
        aggregate_selected,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    future_robust = evaluate_selected_profiles(
        robust_selected,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    future_comparison = pd.concat(
        [_tag(future_aggregate, "aggregate_validation"), _tag(future_robust, "robust_validation_windows")],
        ignore_index=True,
    )
    future_comparison.to_csv(out_dir / "future_selection_frontier.csv", index=False)

    aggregate_windows = evaluate_profile_windows(
        aggregate_selected,
        test.step,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    robust_windows = evaluate_profile_windows(
        robust_selected,
        test.step,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    pd.concat(
        [_tag(aggregate_windows, "aggregate_validation"), _tag(robust_windows, "robust_validation_windows")],
        ignore_index=True,
    ).to_csv(out_dir / "future_selection_windows_50_per_10k.csv", index=False)

    selection_rows = []
    for selection, frame in (
        ("aggregate_validation", aggregate_selected),
        ("robust_validation_windows", robust_selected),
    ):
        for _, row in frame.iterrows():
            selection_rows.append({
                "selection": selection,
                "profile": str(row.profile),
                "alpha": float(row.alpha),
                "amount_scale": float(row.amount_scale),
            })
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(out_dir / "selection_comparison.csv", index=False)

    future_50 = future_comparison.loc[future_comparison.target_alerts_per_10k == reference_capacity]
    future_summary = {}
    for _, row in future_50.iterrows():
        key = f"{row.selection}:{row.profile}"
        future_summary[key] = {
            "alpha": float(row.alpha),
            "precision": float(row.precision),
            "fraud_recall": float(row.recall),
            "fraud_value_recall": float(row.fraud_value_recall),
            "legitimate_alerts_per_10k": float(row.legit_alerts_per_10k),
        }

    summary = {
        "audit": audit,
        "split": {"train_cut": split.train_cut, "validation_cut": split.validation_cut},
        "model": "transaction_plus_relational",
        "selection_capacity_alerts_per_10k": reference_capacity,
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "validation_windows": 3,
        "robust_selection_rule": {
            "case_first": "max worst-window case recall, then mean recall/precision/value recall",
            "balanced": "max worst-window case/value harmonic mean, then mean harmonic mean/precision",
            "value_first": "max worst-window fraud-value recall, then mean value recall/precision/case recall",
            "final_tie_break": "lower alpha",
        },
        "selection_comparison": selection_rows,
        "future_reference_50_per_10k": future_summary,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "Both aggregate and robust routing profiles use validation labels only; future labels never select alpha.",
            "Robust selection prioritises the weakest contiguous validation window before average validation performance.",
            "The amount scale is fitted once on the full validation period and frozen for all validation windows and future evaluation.",
            "A robust profile can trade average validation performance for temporal stability; this is a routing-policy choice, not a new predictive model.",
            "All comparisons use the same exact review capacities and stable non-label event-key tie-breaker.",
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
