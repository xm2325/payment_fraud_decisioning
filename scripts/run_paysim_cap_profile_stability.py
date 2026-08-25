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
from fraud_decisioning.paysim_backlog_cap import continuous_rescore_bounded_backlog
from fraud_decisioning.paysim_backlog_handoff import queue_metrics
from fraud_decisioning.paysim_cap_profile_stability import (
    PROFILE_GRID,
    profile_stability_summary,
)
from fraud_decisioning.paysim_cap_stability import (
    CAPACITY_GRID,
    TARGET_CAP,
    cohort_queue_comparison,
    pathwise_queue_comparison,
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


POLICY_SELECTION_CAPACITY = 50
TEST_WINDOW_STEPS = 50
POLICY_WINDOWS = 3


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
        alerts_per_10k=POLICY_SELECTION_CAPACITY,
        n_windows=POLICY_WINDOWS,
        alphas=DEFAULT_ALPHA_GRID,
    )
    selected = select_robust_profiles(robustness_summary(policy_grid))
    selected["selection_split"] = "cycle_policy_windows_only"
    return {"model": model, "calibrator": calibrator, "selected": selected}


def _profile(bundle, profile: str):
    row = bundle["selected"].loc[bundle["selected"].profile == profile]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one selected {profile} row")
    return row.iloc[0]


def _probability(bundle, frame) -> np.ndarray:
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    return calibrate(bundle["calibrator"], raw)


def _validate_v16_balanced_reference(
    pathwise: pd.DataFrame,
    metrics: pd.DataFrame,
    reference_pathwise: Path,
    reference_metrics: Path,
) -> bool:
    if not reference_pathwise.exists() or not reference_metrics.exists():
        return False
    ref_path = pd.read_csv(reference_pathwise)
    ref_metrics = pd.read_csv(reference_metrics)

    current_path = pathwise.loc[pathwise.profile == "balanced"].copy()
    if len(current_path) != len(CAPACITY_GRID):
        raise ValueError("Expected one balanced pathwise row per v1.16 capacity")
    for capacity in CAPACITY_GRID:
        left = current_path.loc[current_path.capacity_alerts_per_10k == capacity]
        right = ref_path.loc[ref_path.capacity_alerts_per_10k == capacity]
        if len(left) != 1 or len(right) != 1:
            raise ValueError(f"Missing v1.16 pathwise reference for capacity {capacity}")
        left = left.iloc[0]
        right = right.iloc[0]
        for column in (
            "reference_alerts",
            "candidate_alerts",
            "final_overlap",
            "final_jaccard",
            "reference_only",
            "candidate_only",
            "exact_final_queue",
            "common_review_step_equal",
            "exact_review_path",
            "divergent_review_steps",
            "min_cumulative_jaccard",
            "infinite_refresh_rescored",
            "capped_refresh_rescored",
            "refresh_rescore_reduction",
        ):
            if isinstance(right[column], (bool, np.bool_)) or str(right[column]) in {"True", "False"}:
                if str(left[column]).lower() != str(right[column]).lower():
                    raise AssertionError(
                        f"balanced capacity {capacity} failed v1.16 reproduction for {column}"
                    )
            elif not np.isclose(float(left[column]), float(right[column]), rtol=0.0, atol=1e-12):
                raise AssertionError(
                    f"balanced capacity {capacity} failed v1.16 reproduction for {column}"
                )

        for cap_name in ("5000", "infinite"):
            left_metric = metrics.loc[
                (metrics.profile == "balanced")
                & (metrics.capacity_alerts_per_10k == capacity)
                & (metrics.max_pending_cases.astype(str) == cap_name)
            ]
            right_metric = ref_metrics.loc[
                (ref_metrics.capacity_alerts_per_10k == capacity)
                & (ref_metrics.max_pending_cases.astype(str) == cap_name)
            ]
            if len(left_metric) != 1 or len(right_metric) != 1:
                raise ValueError(
                    f"Missing v1.16 balanced metric reference for capacity {capacity}, cap {cap_name}"
                )
            left_metric = left_metric.iloc[0]
            right_metric = right_metric.iloc[0]
            for column in (
                "alerts",
                "precision",
                "fraud_recall",
                "fraud_value_recall",
                "balanced_hmean",
                "review_delay_mean_steps",
                "review_delay_p90_steps",
                "review_delay_max_steps",
                "total_refresh_rescored",
                "max_refresh_rescored",
            ):
                if not np.isclose(
                    float(left_metric[column]), float(right_metric[column]), rtol=0.0, atol=1e-12
                ):
                    raise AssertionError(
                        f"balanced capacity {capacity}, cap {cap_name} failed v1.16 reproduction for {column}"
                    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_cap_profile_stability")
    parser.add_argument(
        "--v16-pathwise-reference",
        default="results/paysim_cap_stability/capacity_pathwise_stability.csv",
    )
    parser.add_argument(
        "--v16-metrics-reference",
        default="results/paysim_cap_stability/capacity_queue_metrics.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_cap_profile_stability_features.parquet"
    db = out_dir / "paysim_cap_profile_stability.duckdb"
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
        raise ValueError("Canonical profile-stability audit expects three rolling cycles")

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
    selected_profiles = pd.concat(selected_rows, ignore_index=True)
    selected_profiles.to_csv(out_dir / "selected_profiles_by_cycle.csv", index=False)

    for cycle_id, bundle in bundles.items():
        available = set(bundle["selected"].profile.astype(str))
        missing = set(PROFILE_GRID).difference(available)
        if missing:
            raise ValueError(f"cycle {cycle_id} missing predeclared profiles: {sorted(missing)}")

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

    probability_by_regime = {
        cycle_id: _probability(bundle, future) for cycle_id, bundle in bundles.items()
    }
    score_by_profile: dict[str, dict[int, np.ndarray]] = {}
    for profile in PROFILE_GRID:
        profile_scores = {}
        for cycle_id, bundle in bundles.items():
            row = _profile(bundle, profile)
            profile_scores[cycle_id] = priority_score(
                probability_by_regime[cycle_id],
                future.amount,
                float(row.alpha),
                float(row.amount_scale),
            )
        score_by_profile[profile] = profile_scores

    metric_rows = []
    pathwise_rows = []
    cohort_rows = []
    refresh_frames = []

    for profile in PROFILE_GRID:
        for capacity in CAPACITY_GRID:
            runs = {}
            for cap_name, cap_value in (("infinite", None), (str(TARGET_CAP), TARGET_CAP)):
                selected, review_step, review_regime, evicted, schedule, refresh = (
                    continuous_rescore_bounded_backlog(
                        future_step,
                        future_key,
                        regime_by_step,
                        score_by_profile[profile],
                        alerts_per_10k=float(capacity),
                        max_pending_cases=cap_value,
                    )
                )
                metrics = queue_metrics(future_step, y, amount, selected, review_step)
                total_rescored = int(refresh.pending_rescored.sum()) if not refresh.empty else 0
                max_rescored = int(refresh.pending_rescored.max()) if not refresh.empty else 0
                metric_rows.append(
                    {
                        "profile": profile,
                        "capacity_alerts_per_10k": int(capacity),
                        "max_pending_cases": cap_name,
                        **metrics,
                        "total_refresh_rescored": total_rescored,
                        "max_refresh_rescored": max_rescored,
                        "peak_pending_cases": int(schedule.pending_after_selection_and_cap.max()),
                        "evicted_cases": int(evicted.sum()),
                    }
                )
                if not refresh.empty:
                    refresh = refresh.copy()
                    refresh.insert(0, "profile", profile)
                    refresh.insert(1, "capacity_alerts_per_10k", int(capacity))
                    refresh_frames.append(refresh)
                runs[cap_name] = {
                    "selected": selected,
                    "review_step": review_step,
                    "total_rescored": total_rescored,
                }

            comparison = pathwise_queue_comparison(
                future_step,
                runs["infinite"]["selected"],
                runs["infinite"]["review_step"],
                runs[str(TARGET_CAP)]["selected"],
                runs[str(TARGET_CAP)]["review_step"],
            )
            infinite_rescored = int(runs["infinite"]["total_rescored"])
            capped_rescored = int(runs[str(TARGET_CAP)]["total_rescored"])
            pathwise_rows.append(
                {
                    "profile": profile,
                    "capacity_alerts_per_10k": int(capacity),
                    "candidate_cap": int(TARGET_CAP),
                    **comparison,
                    "infinite_refresh_rescored": infinite_rescored,
                    "capped_refresh_rescored": capped_rescored,
                    "refresh_rescore_reduction": (
                        float(1.0 - capped_rescored / infinite_rescored)
                        if infinite_rescored > 0
                        else 0.0
                    ),
                    "stability_class": (
                        "PATHWISE_PRESERVED"
                        if bool(comparison["exact_review_path"])
                        else "PATH_DIVERGED"
                    ),
                }
            )

            for cycle in cycles:
                cohort = cohort_queue_comparison(
                    future_step,
                    runs["infinite"]["selected"],
                    runs[str(TARGET_CAP)]["selected"],
                    step_min=int(cycle.test_step_min),
                    step_max=int(cycle.test_step_max),
                )
                cohort_rows.append(
                    {
                        "profile": profile,
                        "capacity_alerts_per_10k": int(capacity),
                        "arrival_cycle": int(cycle.cycle),
                        "arrival_step_min": int(cycle.test_step_min),
                        "arrival_step_max": int(cycle.test_step_max),
                        **cohort,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["profile", "capacity_alerts_per_10k", "max_pending_cases"]
    )
    pathwise = pd.DataFrame(pathwise_rows).sort_values(
        ["profile", "capacity_alerts_per_10k"]
    )
    cohorts = pd.DataFrame(cohort_rows).sort_values(
        ["profile", "capacity_alerts_per_10k", "arrival_cycle"]
    )
    profile_summary = profile_stability_summary(pathwise)

    metrics.to_csv(out_dir / "profile_capacity_queue_metrics.csv", index=False)
    pathwise.to_csv(out_dir / "profile_capacity_pathwise_stability.csv", index=False)
    cohorts.to_csv(out_dir / "profile_capacity_cohort_stability.csv", index=False)
    profile_summary.to_csv(out_dir / "profile_stability_summary.csv", index=False)
    pd.concat(refresh_frames, ignore_index=True).to_csv(
        out_dir / "profile_capacity_refresh_workload.csv", index=False
    )

    reference_reproduced = _validate_v16_balanced_reference(
        pathwise,
        metrics,
        Path(args.v16_pathwise_reference),
        Path(args.v16_metrics_reference),
    )
    all_preserved = bool(profile_summary.all_tested_capacities_pathwise_preserved.all())
    summary = {
        "audit": audit,
        "profiles": list(PROFILE_GRID),
        "policy_selection_capacity_alerts_per_10k": POLICY_SELECTION_CAPACITY,
        "stress_capacity_grid_alerts_per_10k": list(CAPACITY_GRID),
        "candidate_cap": TARGET_CAP,
        "continuous_future_step_min": int(future_step.min()),
        "continuous_future_step_max": int(future_step.max()),
        "continuous_future_n": int(len(future)),
        "refresh_steps": [int(cycles[1].test_step_min), int(cycles[2].test_step_min)],
        "v1_16_balanced_reference_reproduced": bool(reference_reproduced),
        "all_profiles_all_capacities_pathwise_preserved": all_preserved,
        "profile_summary": profile_summary.to_dict(orient="records"),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The 5,000-case cap, the three predeclared Fraud Ops profiles and the 10/25/50/100 reviews-per-10k stress grid are fixed before reading v1.17 full-data results.",
            "Each profile uses only its robust policy-window-selected alpha and amount scale for the corresponding rolling cycle; v1.17 does not re-select policy parameters from the test horizon.",
            "PATHWISE_PRESERVED requires the same final reviewed cases and the same review step for every reviewed case within each profile/capacity combination.",
            "The balanced rows must reproduce the committed v1.16 pathwise and queue-metric references when those files are available in PR/main context.",
            "Cross-profile preservation is queue-mechanics evidence on this benchmark, not proof that a 5,000-case cap is sufficient for other traffic, models or production operations.",
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
