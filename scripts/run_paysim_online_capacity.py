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
from fraud_decisioning.paysim_online_capacity import (
    batch_vs_backlog_capacity,
    batch_vs_online_capacity,
)
from fraud_decisioning.paysim_routing_profiles import DEFAULT_ALPHA_GRID, priority_score
from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)
from fraud_decisioning.paysim_stage_separation import (
    split_validation_stages,
    stage_masks,
)


BUDGETS = (10, 25, 50, 100)
REFERENCE_CAPACITY = 50
CAUSAL_COMPARATORS = (
    ("current_step_only", batch_vs_online_capacity),
    ("seen_so_far_backlog", batch_vs_backlog_capacity),
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


def _reference_metrics(row) -> dict:
    keys = (
        "batch_precision",
        "online_precision",
        "delta_precision",
        "batch_fraud_recall",
        "online_fraud_recall",
        "delta_fraud_recall",
        "batch_fraud_value_recall",
        "online_fraud_value_recall",
        "delta_fraud_value_recall",
        "queue_overlap_rate",
        "queue_jaccard",
        "replacement_rate",
        "review_delay_mean_steps",
        "review_delay_p90_steps",
        "review_delay_max_steps",
        "fraud_review_delay_mean_steps",
        "fraud_review_delay_p90_steps",
    )
    output = {
        "alerts": int(row.online_alerts),
        "incremental_fraud_cases_from_online_swaps": int(
            row.incremental_fraud_cases_from_online_swaps
        ),
        "incremental_fraud_value_from_online_swaps": float(
            row.incremental_fraud_value_from_online_swaps
        ),
    }
    for key in keys:
        output[key] = float(getattr(row, key))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_online_capacity")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_online_capacity_features.parquet"
    db = out_dir / "paysim_online_capacity.duckdb"
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

    stage_split = split_validation_stages(validation.step, calibration_fraction=0.5)
    calibration_mask, policy_mask = stage_masks(validation.step, stage_split)
    calibration = validation.loc[calibration_mask].copy()
    policy = validation.loc[policy_mask].copy()

    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = fit_sigmoid_calibrator(calibration_raw, calibration.is_fraud)
    policy_probability = calibrate(
        calibrator, model.predict_proba(policy[features])[:, 1]
    )
    future_probability = calibrate(
        calibrator, model.predict_proba(future[features])[:, 1]
    )

    window_grid = validation_window_alpha_grid(
        policy.step,
        policy.is_fraud,
        policy_probability,
        policy.amount,
        policy.event_key,
        alerts_per_10k=REFERENCE_CAPACITY,
        n_windows=3,
        alphas=DEFAULT_ALPHA_GRID,
    )
    selected = select_robust_profiles(robustness_summary(window_grid))
    selected["selection_split"] = "policy_selection_windows_only"
    selected.to_csv(out_dir / "selected_profiles.csv", index=False)

    comparison_rows: list[dict] = []
    schedule_frames: list[pd.DataFrame] = []

    for _, profile_row in selected.iterrows():
        profile = str(profile_row.profile)
        alpha = float(profile_row.alpha)
        amount_scale = float(profile_row.amount_scale)
        score = priority_score(future_probability, future.amount, alpha, amount_scale)

        for budget in BUDGETS:
            for declared_contract, comparator in CAUSAL_COMPARATORS:
                row, schedule = comparator(
                    future.step,
                    future.is_fraud,
                    score,
                    future.amount,
                    future.event_key,
                    alerts_per_10k=float(budget),
                )
                if row["routing_contract"] != declared_contract:
                    raise AssertionError("Comparator routing contract changed unexpectedly")
                row.update(
                    {
                        "profile": profile,
                        "alpha": alpha,
                        "amount_scale": amount_scale,
                        "future_step_min": int(future.step.min()),
                        "future_step_max": int(future.step.max()),
                    }
                )
                comparison_rows.append(row)

                if budget == REFERENCE_CAPACITY:
                    schedule = schedule.copy()
                    schedule.insert(0, "profile", profile)
                    schedule.insert(1, "alpha", alpha)
                    schedule.insert(2, "routing_contract", declared_contract)
                    schedule.insert(3, "target_alerts_per_10k", float(budget))
                    schedule_frames.append(schedule)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "batch_vs_online_frontier.csv", index=False)
    schedules = pd.concat(schedule_frames, ignore_index=True)
    schedules.to_csv(out_dir / "online_schedule_50_per_10k.csv", index=False)

    reference = comparison.loc[
        comparison.target_alerts_per_10k == REFERENCE_CAPACITY
    ].copy()
    reference_rows: dict[str, dict] = {}
    for profile in sorted(reference.profile.unique()):
        profile_rows = reference.loc[reference.profile == profile]
        profile_result: dict[str, dict] = {
            "alpha": float(profile_rows.iloc[0].alpha),
        }
        for contract in sorted(profile_rows.routing_contract.unique()):
            row = profile_rows.loc[profile_rows.routing_contract == contract].iloc[0]
            profile_result[str(contract)] = _reference_metrics(row)
        reference_rows[str(profile)] = profile_result

    final_schedule = schedules.groupby(
        ["profile", "routing_contract"], as_index=False
    ).tail(1)
    expected_final = int(np.floor(REFERENCE_CAPACITY * len(future) / 10_000))
    for _, row in final_schedule.iterrows():
        if int(row.selected_cumulative) != expected_final:
            raise AssertionError("Every causal reference must consume identical final capacity")

    summary = {
        "audit": audit,
        "outer_split": {
            "train_cut": split.train_cut,
            "future_cut": split.validation_cut,
        },
        "future_step_min": int(future.step.min()),
        "future_step_max": int(future.step.max()),
        "future_n": int(len(future)),
        "model": "transaction_plus_relational_fixed_before_online_capacity_audit",
        "policy_selection": "v1.7 disjoint calibration/policy stages with robust alpha selection",
        "budgets_per_10k": list(BUDGETS),
        "causal_routing_contracts": [name for name, _ in CAUSAL_COMPARATORS],
        "reference_capacity_alerts_per_10k": REFERENCE_CAPACITY,
        "reference_50_per_10k": reference_rows,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The existing whole-window exact-capacity metric is retained as a retrospective batch benchmark, not an online routing claim.",
            "Both causal rules process PaySim steps in ascending order and never compare an observed transaction with a later-step score.",
            "The current_step_only rule immediately spends newly earned capacity within the current step; it is a strict low-latency micro-batch lower comparator rather than the only plausible operations design.",
            "The seen_so_far_backlog rule lets earlier unreviewed transactions remain eligible for later capacity, so it can defer reviews while still using only information available by each decision step.",
            "Capacity entitlement accrues as floor(alerts_per_10k * cumulative_transactions / 10000); fractional capacity carries forward and all comparators use exactly the same final number of reviews.",
            "Within one PaySim step, transactions are treated as one micro-batch because this benchmark does not define a finer causal ordering contract for the audit.",
            "Model training, probability calibration and alpha selection remain frozen before future steps 595-743; future labels do not choose the policy.",
            "PaySim is synthetic mobile-money data; synthetic transaction amount is not interpreted as prevented loss or production financial impact."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
