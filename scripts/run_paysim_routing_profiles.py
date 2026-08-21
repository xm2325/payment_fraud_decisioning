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
    evaluate_profile_windows,
    evaluate_selected_profiles,
    select_profiles,
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
    parser.add_argument("--out", default="results/paysim_routing_profiles")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_routing_features.parquet"
    db = out_dir / "paysim_routing.duckdb"
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
    validation_grid = alpha_grid_metrics(
        val.is_fraud,
        val_probability,
        val.amount,
        val.event_key,
        alerts_per_10k=reference_capacity,
        alphas=DEFAULT_ALPHA_GRID,
    )
    validation_grid.to_csv(out_dir / "validation_alpha_grid_50_per_10k.csv", index=False)

    selected = select_profiles(validation_grid)
    selected.to_csv(out_dir / "selected_routing_profiles.csv", index=False)

    future_profiles = evaluate_selected_profiles(
        selected,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )
    future_profiles.to_csv(out_dir / "future_profile_frontier.csv", index=False)

    # Full future alpha grid is sensitivity-only; profile selection above never uses it.
    future_alpha_grid = alpha_grid_metrics(
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        alerts_per_10k=reference_capacity,
        alphas=DEFAULT_ALPHA_GRID,
        amount_scale=float(selected.amount_scale.iloc[0]),
    )
    future_alpha_grid.to_csv(out_dir / "future_alpha_grid_diagnostic.csv", index=False)

    windows = evaluate_profile_windows(
        selected,
        test.step,
        test.is_fraud,
        future_probability,
        test.amount,
        test.event_key,
        alerts_per_10k=reference_capacity,
        n_windows=3,
    )
    windows.to_csv(out_dir / "future_profile_windows_50_per_10k.csv", index=False)

    reference_rows = future_profiles.loc[
        future_profiles.target_alerts_per_10k == reference_capacity
    ]
    summary_profiles = {}
    for _, row in reference_rows.iterrows():
        summary_profiles[str(row.profile)] = {
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
        "selection_rule": {
            "case_first": "max validation fraud-case recall, then precision/value recall",
            "balanced": "max validation harmonic mean of case recall and fraud-value recall, then precision",
            "value_first": "max validation fraud-value recall, then precision/case recall",
            "final_tie_break": "lower alpha",
        },
        "selected_profiles": selected[["profile", "alpha", "recall", "fraud_value_recall", "precision"]].to_dict("records"),
        "future_reference_50_per_10k": summary_profiles,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "Alpha values are pre-specified before future-test evaluation; selected profiles use validation labels only.",
            "Alpha=0 is pure fraud-probability ranking and alpha=1 is probability-times-amount expected-loss-style prioritisation.",
            "Intermediate alpha values are routing-policy trade-offs, not new predictive model features.",
            "The balanced profile uses an explicit validation objective rather than tuning alpha on future performance.",
            "All profiles use the same exact review capacity and stable non-label event-key tie-breaker.",
            "Future alpha-grid results are diagnostic sensitivity only and cannot be used to re-select the profile.",
            "PaySim is synthetic mobile-money data; results are benchmark evidence, not production impact or prevented loss."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
