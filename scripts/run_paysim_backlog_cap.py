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
from fraud_decisioning.paysim_backlog_cap import (
    CAP_GRID,
    cap_label,
    continuous_rescore_bounded_backlog,
    eviction_metrics,
)
from fraud_decisioning.paysim_backlog_handoff import queue_metrics, queue_overlap
from fraud_decisioning.paysim_features import FEATURE_SETS
from fraud_decisioning.paysim_full import (
    _load_split,
    audit_sql,
    connect_duckdb,
    determine_split,
    materialise_features,
    validate_canonical,
)
from fraud_decisioning.paysim_rolling_refresh import (
    build_rolling_cycles,
    cycle_contract_frame,
    validate_cycle_sequence,
)
from fraud_decisioning.paysim_routing_profiles import DEFAULT_ALPHA_GRID, priority_score
from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)
from fraud_decisioning.paysim_stage_separation import split_validation_stages


REFERENCE_CAPACITY = 50
TEST_WINDOW_STEPS = 50
POLICY_WINDOWS = 3
PROFILE = "balanced"
DETECTION_NONINFERIORITY_MARGIN = 0.02
WORKLOAD_REDUCTION_TARGET = 0.50


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


def _where(lo: int, hi: int) -> str:
    return f"step >= {int(lo)} AND step <= {int(hi)}"


def _fit_cycle_bundle(train, calibration, policy):
    features = FEATURE_SETS["transaction_plus_relational"]
    model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = fit_sigmoid_calibrator(calibration_raw, calibration.is_fraud)
    policy_probability = calibrate(
        calibrator, model.predict_proba(policy[features])[:, 1]
    )
    policy_grid = validation_window_alpha_grid(
        policy.step,
        policy.is_fraud,
        policy_probability,
        policy.amount,
        policy.event_key,
        alerts_per_10k=REFERENCE_CAPACITY,
        n_windows=POLICY_WINDOWS,
        alphas=DEFAULT_ALPHA_GRID,
    )
    selected = select_robust_profiles(robustness_summary(policy_grid))
    selected["selection_split"] = "cycle_policy_windows_only"
    return {"model": model, "calibrator": calibrator, "selected": selected}


def _profile(bundle):
    row = bundle["selected"].loc[bundle["selected"].profile == PROFILE]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one selected {PROFILE} row")
    return row.iloc[0]


def _score(bundle, frame) -> np.ndarray:
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    probability = calibrate(bundle["calibrator"], raw)
    profile_row = _profile(bundle)
    return priority_score(
        probability,
        frame.amount,
        float(profile_row.alpha),
        float(profile_row.amount_scale),
    )


def _cohort_rows(future, cycles, selected, review_step, cap_name: str):
    rows = []
    step = future.step.to_numpy(dtype=int)
    for cycle in cycles:
        mask = (step >= cycle.test_step_min) & (step <= cycle.test_step_max)
        rows.append(
            {
                "max_pending_cases": cap_name,
                "arrival_cycle": int(cycle.cycle),
                "arrival_step_min": int(cycle.test_step_min),
                "arrival_step_max": int(cycle.test_step_max),
                "transactions": int(mask.sum()),
                **queue_metrics(
                    step[mask],
                    future.is_fraud.to_numpy()[mask],
                    future.amount.to_numpy()[mask],
                    selected[mask],
                    review_step[mask],
                ),
            }
        )
    return rows


def _validate_v13_infinite(frontier: pd.DataFrame, reference_path: Path) -> bool:
    if not reference_path.exists():
        return False
    reference = pd.read_csv(reference_path)
    reference = reference.loc[reference.strategy == "rescore_pending"]
    if len(reference) != 1:
        raise ValueError("Expected one v1.13 rescore_pending reference row")
    infinite = frontier.loc[frontier.max_pending_cases == "infinite"]
    if len(infinite) != 1:
        raise ValueError("Expected one infinite-cap row")
    left = reference.iloc[0]
    right = infinite.iloc[0]
    for column in (
        "alerts",
        "precision",
        "fraud_recall",
        "fraud_value_recall",
        "balanced_hmean",
        "review_delay_mean_steps",
        "review_delay_p90_steps",
        "review_delay_max_steps",
    ):
        if not np.isclose(float(left[column]), float(right[column]), rtol=0.0, atol=1e-12):
            raise AssertionError(f"Infinite cap failed to reproduce v1.13 column {column}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_backlog_cap")
    parser.add_argument(
        "--handoff-reference",
        default="results/paysim_backlog_handoff/strategy_metrics.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_backlog_cap_features.parquet"
    db = out_dir / "paysim_backlog_cap.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {
        key: (float(value) if key == "fraud_rate" else int(value))
        for key, value in audit_row.items()
    }
    validate_canonical(audit)
    outer_split = determine_split(con, args.parquet_glob)
    materialise_features(con, args.parquet_glob, work)

    validation = _load_split(
        con,
        work,
        f"step >= {outer_split.train_cut} AND step < {outer_split.validation_cut}",
    )
    stage_split = split_validation_stages(validation.step, calibration_fraction=0.5)
    work_sql = str(work).replace("'", "''")
    all_steps = con.execute(
        f"SELECT DISTINCT step::INTEGER AS step FROM read_parquet('{work_sql}') ORDER BY step"
    ).fetchnumpy()["step"]
    cycles = build_rolling_cycles(
        all_steps,
        initial_test_step=outer_split.validation_cut,
        calibration_n_steps=stage_split.calibration_n_steps,
        policy_n_steps=stage_split.policy_n_steps,
        test_window_n_steps=TEST_WINDOW_STEPS,
    )
    validate_cycle_sequence(cycles)
    cycle_contract_frame(cycles).to_csv(out_dir / "cycle_contract.csv", index=False)
    del validation
    if len(cycles) != 3:
        raise ValueError("Canonical bounded-backlog frontier expects three rolling cycles")

    bundles = {}
    selected_rows = []
    for cycle in cycles:
        train = _load_split(con, work, _where(cycle.train_step_min, cycle.train_step_max))
        calibration = _load_split(
            con, work, _where(cycle.calibration_step_min, cycle.calibration_step_max)
        )
        policy = _load_split(con, work, _where(cycle.policy_step_min, cycle.policy_step_max))
        bundle = _fit_cycle_bundle(train, calibration, policy)
        bundles[int(cycle.cycle)] = bundle
        selected = bundle["selected"].copy()
        selected.insert(0, "cycle", int(cycle.cycle))
        selected_rows.append(selected)
        del train, calibration, policy
    pd.concat(selected_rows, ignore_index=True).to_csv(
        out_dir / "selected_profiles_by_cycle.csv", index=False
    )

    future = _load_split(
        con,
        work,
        f"step >= {cycles[0].test_step_min} AND step <= {cycles[-1].test_step_max}",
    )
    future_step = future.step.to_numpy(dtype=int)
    future_key = future.event_key.to_numpy(dtype=np.uint64)
    y = future.is_fraud.to_numpy(dtype=int)
    amount = future.amount.to_numpy(dtype=float)

    regime_by_step = {}
    for cycle in cycles:
        for value in range(int(cycle.test_step_min), int(cycle.test_step_max) + 1):
            regime_by_step[value] = int(cycle.cycle)
    score_by_regime = {
        cycle_id: _score(bundle, future) for cycle_id, bundle in bundles.items()
    }

    masks = {}
    metric_rows = []
    schedule_frames = []
    refresh_frames = []
    cohort_rows = []

    for cap in CAP_GRID:
        name = cap_label(cap)
        selected, review_step, review_regime, evicted, schedule, refresh = (
            continuous_rescore_bounded_backlog(
                future_step,
                future_key,
                regime_by_step,
                score_by_regime,
                alerts_per_10k=REFERENCE_CAPACITY,
                max_pending_cases=cap,
            )
        )
        masks[name] = selected
        metrics = queue_metrics(future_step, y, amount, selected, review_step)
        eviction = eviction_metrics(y, amount, evicted)
        total_rescored = int(refresh.pending_rescored.sum()) if not refresh.empty else 0
        max_rescored = int(refresh.pending_rescored.max()) if not refresh.empty else 0
        peak_pending = int(schedule.pending_after_selection_and_cap.max())
        metric_rows.append(
            {
                "max_pending_cases": name,
                "cap_numeric": np.inf if cap is None else int(cap),
                "profile": PROFILE,
                **metrics,
                **eviction,
                "total_refresh_rescored": total_rescored,
                "max_refresh_rescored": max_rescored,
                "peak_pending_cases": peak_pending,
            }
        )
        schedule_frames.append(schedule)
        if not refresh.empty:
            refresh_frames.append(refresh)
        cohort_rows.extend(_cohort_rows(future, cycles, selected, review_step, name))

    frontier = pd.DataFrame(metric_rows)
    infinite = frontier.loc[frontier.max_pending_cases == "infinite"].iloc[0]
    for metric in ("precision", "fraud_recall", "fraud_value_recall"):
        frontier[f"delta_{metric}_vs_infinite"] = frontier[metric] - float(infinite[metric])
        frontier[f"{metric}_guardrail_pass"] = (
            frontier[f"delta_{metric}_vs_infinite"] >= -DETECTION_NONINFERIORITY_MARGIN
        )
    infinite_rescored = float(infinite.total_refresh_rescored)
    frontier["refresh_rescore_reduction_vs_infinite"] = (
        1.0 - frontier.total_refresh_rescored / infinite_rescored
        if infinite_rescored > 0
        else 0.0
    )
    frontier["detection_guardrails_pass"] = (
        frontier.precision_guardrail_pass
        & frontier.fraud_recall_guardrail_pass
        & frontier.fraud_value_recall_guardrail_pass
    )
    frontier["workload_reduction_target_pass"] = (
        frontier.refresh_rescore_reduction_vs_infinite >= WORKLOAD_REDUCTION_TARGET
    )
    frontier["operational_candidate"] = (
        frontier.detection_guardrails_pass & frontier.workload_reduction_target_pass
    )
    frontier = frontier.sort_values("cap_numeric")
    frontier.to_csv(out_dir / "cap_frontier.csv", index=False)

    pd.concat(schedule_frames, ignore_index=True).to_csv(
        out_dir / "cap_capacity_schedule.csv", index=False
    )
    pd.concat(refresh_frames, ignore_index=True).to_csv(
        out_dir / "cap_refresh_workload.csv", index=False
    )
    pd.DataFrame(cohort_rows).to_csv(out_dir / "cap_arrival_cohort_metrics.csv", index=False)

    overlap_rows = []
    for name, mask in masks.items():
        if name == "infinite":
            continue
        overlap_rows.append(
            {
                "reference_cap": "infinite",
                "max_pending_cases": name,
                **queue_overlap(masks["infinite"], mask),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(out_dir / "cap_queue_overlap.csv", index=False)

    entitlement = int(np.floor(REFERENCE_CAPACITY * len(future) / 10_000))
    if not (frontier.alerts == entitlement).all():
        raise AssertionError("Every backlog cap must consume the same continuous final capacity")

    reference_reproduced = _validate_v13_infinite(
        frontier, Path(args.handoff_reference)
    )
    candidate_rows = frontier.loc[frontier.operational_candidate]
    summary = {
        "audit": audit,
        "profile": PROFILE,
        "reference_capacity_alerts_per_10k": REFERENCE_CAPACITY,
        "continuous_future_step_min": int(future_step.min()),
        "continuous_future_step_max": int(future_step.max()),
        "continuous_future_n": int(len(future)),
        "continuous_final_review_entitlement": entitlement,
        "refresh_steps": [int(cycles[1].test_step_min), int(cycles[2].test_step_min)],
        "cap_grid": [cap_label(value) for value in CAP_GRID],
        "detection_noninferiority_margin": DETECTION_NONINFERIORITY_MARGIN,
        "workload_reduction_target": WORKLOAD_REDUCTION_TARGET,
        "v1_13_infinite_reference_reproduced": bool(reference_reproduced),
        "operational_candidate_caps": candidate_rows.max_pending_cases.tolist(),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The bounded backlog evicts only seen-but-unreviewed cases using the current policy score; fraud labels do not drive eviction or queue selection.",
            "Newly earned analyst capacity is allocated before the pending-memory cap is enforced, so a new arrival can be reviewed even when the retained backlog was full before that step.",
            "When the pool exceeds its cap, the lowest current-score cases are evicted; equal-score eviction preserves the same deterministic event-key priority used by review selection.",
            "At each model refresh, every surviving pending case is rescored by the newly released system. The cap therefore creates a hard upper bound on refresh-time rescore volume per boundary.",
            "All caps use the same balanced rolling systems, continuous 50 reviews/10k entitlement and rescore-on-refresh policy. The only changed control is retained pending-pool size.",
            "The predeclared screen is the same as v1.14: each detection metric may decline by at most 2 percentage points versus infinite backlog and total refresh rescoring must fall by at least 50%.",
            "operational_candidate is a deterministic engineering screen, not statistical evidence or a production recommendation.",
            "The infinite cap must reproduce the v1.13 rescore_pending reference when that main-branch file is available.",
            "PaySim is synthetic; dataset steps are not production SLA units, transaction amount is not prevented loss, and upstream rolling refresh retains the as-of label-availability assumption."
        ],
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
