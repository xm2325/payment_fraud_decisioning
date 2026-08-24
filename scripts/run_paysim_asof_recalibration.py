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
from fraud_decisioning.paysim_asof_recalibration import (
    contiguous_future_windows,
    expanding_recalibration_mask,
    method_summary,
    recalibration_metric_row,
)
from fraud_decisioning.paysim_features import FEATURE_SETS
from fraud_decisioning.paysim_full import (
    _load_split,
    audit_sql,
    connect_duckdb,
    determine_split,
    materialise_features,
    validate_canonical,
)
from fraud_decisioning.paysim_monitoring import ranked_capacity_metrics
from fraud_decisioning.paysim_routing_profiles import amount_scale_from_validation, priority_score
from fraud_decisioning.paysim_stage_separation import split_validation_stages, stage_masks


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
    parser.add_argument("--out", default="results/paysim_asof_recalibration")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_asof_features.parquet"
    db = out_dir / "paysim_asof.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
    validate_canonical(audit)
    split = determine_split(con, args.parquet_glob)
    materialise_features(con, args.parquet_glob, work)

    train = _load_split(con, work, f"step < {split.train_cut}")
    validation = _load_split(con, work, f"step >= {split.train_cut} AND step < {split.validation_cut}")
    future = _load_split(con, work, f"step >= {split.validation_cut}")

    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
    validation_raw = model.predict_proba(validation[features])[:, 1]
    future_raw = model.predict_proba(future[features])[:, 1]

    stage_split = split_validation_stages(validation.step, calibration_fraction=0.5)
    base_mask, policy_mask = stage_masks(validation.step, stage_split)
    base_calibration = validation.loc[base_mask].copy()
    base_raw = validation_raw[base_mask]
    frozen_calibrator = fit_sigmoid_calibrator(base_raw, base_calibration.is_fraud)

    policy = validation.loc[policy_mask].copy()
    amount_scale = amount_scale_from_validation(policy.amount)
    alpha = 0.25
    review_capacity = 50.0

    history = pd.concat([validation, future], ignore_index=True)
    history_raw = np.concatenate([validation_raw, future_raw])
    windows = contiguous_future_windows(future.step, n_windows=3)

    methods = [
        ("frozen_initial", None),
        ("asof_24h", 24),
        ("asof_168h", 168),
        ("instant_history_diagnostic", 0),
    ]
    rows: list[dict] = []

    for window in windows:
        window_mask = (future.step >= window.step_min) & (future.step <= window.step_max)
        window_frame = future.loc[window_mask].copy()
        window_raw = future_raw[np.asarray(window_mask)]

        for method, lag in methods:
            if lag is None:
                calibrator = frozen_calibrator
                calibration_n = int(len(base_calibration))
                calibration_max_step = int(stage_split.calibration_step_max)
                incremental_n = 0
            else:
                calibration_mask, cutoff = expanding_recalibration_mask(
                    history.step,
                    window_start_step=window.step_min,
                    maturity_lag_steps=lag,
                    initial_calibration_step_min=stage_split.calibration_step_min,
                    initial_calibration_step_max=stage_split.calibration_step_max,
                )
                calibration_rows = history.loc[calibration_mask]
                calibration_raw = history_raw[np.asarray(calibration_mask)]
                calibrator = fit_sigmoid_calibrator(calibration_raw, calibration_rows.is_fraud)
                calibration_n = int(calibration_mask.sum())
                calibration_max_step = int(calibration_rows.step.max())
                incremental_n = int((calibration_rows.step > stage_split.calibration_step_max).sum())

            probability = calibrate(calibrator, window_raw)
            row = recalibration_metric_row(
                method=method,
                window=window,
                y=window_frame.is_fraud,
                probability=probability,
                calibration_n=calibration_n,
                calibration_max_step=calibration_max_step,
                maturity_lag_steps=lag,
            )
            row["incremental_refresh_rows"] = incremental_n
            score = priority_score(probability, window_frame.amount, alpha, amount_scale)
            routing = ranked_capacity_metrics(
                window_frame.is_fraud,
                score,
                window_frame.amount,
                window_frame.event_key,
                review_capacity,
            )
            row.update({
                "routing_alpha": alpha,
                "amount_scale": amount_scale,
                "target_alerts_per_10k": review_capacity,
                "routing_precision": float(routing["precision"]),
                "routing_recall": float(routing["recall"]),
                "routing_fraud_value_recall": float(routing["fraud_value_recall"]),
                "routing_legit_alerts_per_10k": float(routing["legit_alerts_per_10k"]),
            })
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "future_window_recalibration.csv", index=False)
    summary_frame = method_summary(results)
    summary_frame.to_csv(out_dir / "recalibration_method_summary.csv", index=False)

    frozen = results.loc[results.method == "frozen_initial"].set_index("window")
    comparisons = []
    for method in ("asof_24h", "asof_168h", "instant_history_diagnostic"):
        current = results.loc[results.method == method].set_index("window")
        for window in frozen.index:
            comparisons.append({
                "method": method,
                "window": window,
                "brier_change_vs_frozen": float(current.loc[window, "brier"] - frozen.loc[window, "brier"]),
                "mean_risk_ratio_change_vs_frozen": float(
                    current.loc[window, "mean_to_observed_ratio"] - frozen.loc[window, "mean_to_observed_ratio"]
                ),
                "routing_recall_change_vs_frozen": float(
                    current.loc[window, "routing_recall"] - frozen.loc[window, "routing_recall"]
                ),
                "routing_value_recall_change_vs_frozen": float(
                    current.loc[window, "routing_fraud_value_recall"] - frozen.loc[window, "routing_fraud_value_recall"]
                ),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(out_dir / "recalibration_vs_frozen.csv", index=False)

    summary = {
        "audit": audit,
        "model": "transaction_plus_relational_fixed",
        "routing_alpha": alpha,
        "routing_amount_scale": float(amount_scale),
        "routing_capacity_alerts_per_10k": review_capacity,
        "initial_calibration_steps": [stage_split.calibration_step_min, stage_split.calibration_step_max],
        "future_windows": [window.__dict__ for window in windows],
        "refresh_methods": {
            "frozen_initial": "never refit after the approved steps 446-519 calibrator",
            "asof_24h": "expanding refit using initial calibration plus post-519 labels matured by 24 steps",
            "asof_168h": "expanding refit using initial calibration plus post-519 labels matured by 168 steps",
            "instant_history_diagnostic": "diagnostic refit using all labels from prior steps; unavailable under delayed-label operation",
        },
        "method_summary": summary_frame.to_dict(orient="records"),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The predictive LightGBM model and alpha=0.25 routing policy are frozen before this experiment.",
            "The initial steps 446-519 calibrator is treated as an already approved deployment artefact; label-lag scenarios govern only later refresh evidence.",
            "A label from step s can enter an as-of refresh before scoring step t only if s + lag < t.",
            "The 24-hour and 168-hour maturity lags are stress-test scenarios, not Moniepoint label-latency estimates.",
            "The instant-history method is diagnostic only under the delayed-label framing.",
            "Recalibration changes absolute probabilities and can also alter amount-weighted queue ordering even when the underlying model ranking is unchanged.",
            "PaySim is synthetic mobile-money data; results are methodological benchmark evidence only."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
