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
PROFILE = "balanced"


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


def _validate_v15_reference(
    metric_frame: pd.DataFrame,
    reference_path: Path,
) -> bool:
    """Require the 50/10k rows to reproduce v1.15 when its main result exists."""
    if not reference_path.exists():
        return False
    reference = pd.read_csv(reference_path)
    checks = (("infinite", "infinite"), (str(TARGET_CAP), str(TARGET_CAP)))
    for current_name, reference_name in checks:
        current = metric_frame.loc[
            (metric_frame.capacity_alerts_per_10k == 50)
            & (metric_frame.max_pending_cases.astype(str) == current_name)
        ]
        ref = reference.loc[reference.max_pending_cases.astype(str) == reference_name]
        if len(current) != 1 or len(ref) != 1:
            raise ValueError(f"Expected one v1.15 reference row for cap {reference_name}")
        left = current.iloc[0]
        right = ref.iloc[0]
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
        ):
            if not np.isclose(float(left[column]), float(right[column]), rtol=0.0, atol=1e-12):
                raise AssertionError(
                    f"v1.16 capacity-50 cap {current_name} failed v1.15 reproduction for {column}"
                )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_cap_stability")
    parser.add_argument(
        "--v15-reference",
        default="results/paysim_backlog_cap/cap_frontier.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_cap_stability_features.parquet"
    db = out_dir / "paysim_cap_stability.duckdb"
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
        raise ValueError("Canonical cap-stability audit expects three rolling cycles")

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

    metric_rows = []
    pathwise_rows = []
    cohort_rows = []
    refresh_frames = []

    for capacity in CAPACITY_GRID:
        runs = {}
        for cap_name, cap_value in (("infinite", None), (str(TARGET_CAP), TARGET_CAP)):
            selected, review_step, review_regime, evicted, schedule, refresh = (
                continuous_rescore_bounded_backlog(
                    future_step,
                    future_key,
                    regime_by_step,
                    score_by_regime,
                    alerts_per_10k=float(capacity),
                    max_pending_cases=cap_value,
                )
            )
            metrics = queue_metrics(future_step, y, amount, selected, review_step)
            total_rescored = int(refresh.pending_rescored.sum()) if not refresh.empty else 0
            max_rescored = int(refresh.pending_rescored.max()) if not refresh.empty else 0
            metric_rows.append(
                {
                    "capacity_alerts_per_10k": int(capacity),
                    "max_pending_cases": cap_name,
                    **metrics,
                    "total_refresh_rescored": total_rescored,
                    "max_refresh_rescored": max_rescored,
                    "peak_pending_cases": int(schedule.pending_after_selection_and_cap.max()),
                    "evicted_cases": int(evicted.sum()),
                }
            )
            refresh = refresh.copy()
            if not refresh.empty:
                refresh.insert(0, "capacity_alerts_per_10k", int(capacity))
                refresh_frames.append(refresh)
            runs[cap_name] = {
                "selected": selected,
                "review_step": review_step,
                "refresh": refresh,
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
                    "capacity_alerts_per_10k": int(capacity),
                    "arrival_cycle": int(cycle.cycle),
                    "arrival_step_min": int(cycle.test_step_min),
                    "arrival_step_max": int(cycle.test_step_max),
                    **cohort,
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["capacity_alerts_per_10k", "max_pending_cases"]
    )
    pathwise = pd.DataFrame(pathwise_rows).sort_values("capacity_alerts_per_10k")
    cohorts = pd.DataFrame(cohort_rows).sort_values(
        ["capacity_alerts_per_10k", "arrival_cycle"]
    )
    metrics.to_csv(out_dir / "capacity_queue_metrics.csv", index=False)
    pathwise.to_csv(out_dir / "capacity_pathwise_stability.csv", index=False)
    cohorts.to_csv(out_dir / "capacity_cohort_stability.csv", index=False)
    pd.concat(refresh_frames, ignore_index=True).to_csv(
        out_dir / "capacity_refresh_workload.csv", index=False
    )

    reference_reproduced = _validate_v15_reference(
        metrics, Path(args.v15_reference)
    )
    preserved = pathwise.loc[
        pathwise.stability_class == "PATHWISE_PRESERVED",
        "capacity_alerts_per_10k",
    ].astype(int).tolist()

    summary = {
        "audit": audit,
        "profile": PROFILE,
        "policy_selection_capacity_alerts_per_10k": POLICY_SELECTION_CAPACITY,
        "stress_capacity_grid_alerts_per_10k": list(CAPACITY_GRID),
        "candidate_cap": TARGET_CAP,
        "continuous_future_step_min": int(future_step.min()),
        "continuous_future_step_max": int(future_step.max()),
        "continuous_future_n": int(len(future)),
        "refresh_steps": [int(cycles[1].test_step_min), int(cycles[2].test_step_min)],
        "v1_15_capacity50_reference_reproduced": bool(reference_reproduced),
        "pathwise_preserved_capacities": preserved,
        "all_predeclared_capacities_pathwise_preserved": bool(
            len(preserved) == len(CAPACITY_GRID)
        ),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "The v1.15 5k cap is not retuned. The stress grid is fixed at 10/25/50/100 reviews per 10k and uses the same balanced rolling systems selected at the original 50-review policy capacity.",
            "PATHWISE_PRESERVED requires the same final reviewed cases and the same review step for every reviewed case; aggregate metric equality alone is insufficient.",
            "Arrival-cohort comparisons localise any final queue displacement to the three existing rolling test periods rather than creating new post-hoc windows.",
            "Capacity stress changes analyst entitlement only; it is not a new policy-selection exercise and does not claim that any tested capacity matches a real production operation.",
            "The 5,000-case cap remains the lowest tested non-zero cap from the predeclared v1.15 grid; this audit does not search below 5,000.",
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
