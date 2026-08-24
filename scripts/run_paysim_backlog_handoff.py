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
from fraud_decisioning.paysim_backlog_handoff import (
    HANDOFF_MODES,
    continuous_backlog_handoff,
    queue_metrics,
    queue_overlap,
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


REFERENCE_CAPACITY = 50
TEST_WINDOW_STEPS = 50
POLICY_WINDOWS = 3
PROFILE = "balanced"
STRATEGIES = ("frozen_incumbent",) + HANDOFF_MODES


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


def _profile(bundle, profile: str = PROFILE):
    row = bundle["selected"].loc[bundle["selected"].profile == profile]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one selected {profile} row")
    return row.iloc[0]


def _score(bundle, frame, profile: str = PROFILE) -> np.ndarray:
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    probability = calibrate(bundle["calibrator"], raw)
    profile_row = _profile(bundle, profile)
    return priority_score(
        probability,
        frame.amount,
        float(profile_row.alpha),
        float(profile_row.amount_scale),
    )


def _arrival_cycle(step: np.ndarray, cycles) -> np.ndarray:
    output = np.full(len(step), -1, dtype=int)
    for cycle in cycles:
        mask = (step >= cycle.test_step_min) & (step <= cycle.test_step_max)
        output[mask] = int(cycle.cycle)
    if np.any(output < 0):
        raise ValueError("Every future transaction must map to one rolling test cycle")
    return output


def _cohort_metrics(future, selected, review_step, strategy: str, cycles) -> list[dict]:
    rows = []
    for cycle in cycles:
        mask = (
            (future.step.to_numpy() >= cycle.test_step_min)
            & (future.step.to_numpy() <= cycle.test_step_max)
        )
        metrics = queue_metrics(
            future.step.to_numpy()[mask],
            future.is_fraud.to_numpy()[mask],
            future.amount.to_numpy()[mask],
            selected[mask],
            review_step[mask],
        )
        rows.append(
            {
                "strategy": strategy,
                "arrival_cycle": int(cycle.cycle),
                "arrival_step_min": int(cycle.test_step_min),
                "arrival_step_max": int(cycle.test_step_max),
                "transactions": int(mask.sum()),
                **metrics,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_backlog_handoff")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_backlog_handoff_features.parquet"
    db = out_dir / "paysim_backlog_handoff.duckdb"
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
        raise ValueError("Canonical backlog handoff audit expects three rolling cycles")

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
    arrival_cycle = _arrival_cycle(future_step, cycles)
    regime_by_step = {}
    for cycle in cycles:
        for value in range(int(cycle.test_step_min), int(cycle.test_step_max) + 1):
            regime_by_step[value] = int(cycle.cycle)

    score_by_regime = {
        cycle_id: _score(bundle, future, PROFILE) for cycle_id, bundle in bundles.items()
    }

    strategy_masks: dict[str, np.ndarray] = {}
    strategy_review_step: dict[str, np.ndarray] = {}
    strategy_review_regime: dict[str, np.ndarray] = {}
    metric_rows = []
    cohort_rows = []
    schedule_frames = []
    refresh_frames = []

    frozen_regime_by_step = {int(value): 1 for value in np.unique(future_step)}
    selected, review_step, review_regime, schedule, refresh = continuous_backlog_handoff(
        future_step,
        future_key,
        frozen_regime_by_step,
        {1: score_by_regime[1]},
        alerts_per_10k=REFERENCE_CAPACITY,
        handoff_mode="retain_old_scores",
    )
    strategy_masks["frozen_incumbent"] = selected
    strategy_review_step["frozen_incumbent"] = review_step
    strategy_review_regime["frozen_incumbent"] = review_regime
    metric_rows.append(
        {
            "strategy": "frozen_incumbent",
            "profile": PROFILE,
            **queue_metrics(
                future_step,
                future.is_fraud.to_numpy(),
                future.amount.to_numpy(),
                selected,
                review_step,
            ),
        }
    )
    cohort_rows.extend(_cohort_metrics(future, selected, review_step, "frozen_incumbent", cycles))
    schedule.insert(0, "strategy", "frozen_incumbent")
    schedule_frames.append(schedule)

    for mode in HANDOFF_MODES:
        selected, review_step, review_regime, schedule, refresh = continuous_backlog_handoff(
            future_step,
            future_key,
            regime_by_step,
            score_by_regime,
            alerts_per_10k=REFERENCE_CAPACITY,
            handoff_mode=mode,
        )
        strategy_masks[mode] = selected
        strategy_review_step[mode] = review_step
        strategy_review_regime[mode] = review_regime
        metric_rows.append(
            {
                "strategy": mode,
                "profile": PROFILE,
                **queue_metrics(
                    future_step,
                    future.is_fraud.to_numpy(),
                    future.amount.to_numpy(),
                    selected,
                    review_step,
                ),
            }
        )
        cohort_rows.extend(_cohort_metrics(future, selected, review_step, mode, cycles))
        schedule.insert(0, "strategy", mode)
        schedule_frames.append(schedule)
        if not refresh.empty:
            refresh.insert(0, "strategy", mode)
            refresh_frames.append(refresh)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "strategy_metrics.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(out_dir / "arrival_cohort_metrics.csv", index=False)
    pd.concat(schedule_frames, ignore_index=True).to_csv(
        out_dir / "continuous_capacity_schedule.csv", index=False
    )
    pd.concat(refresh_frames, ignore_index=True).to_csv(
        out_dir / "refresh_handoff_diagnostics.csv", index=False
    )

    overlap_rows = []
    for reference in ("frozen_incumbent", "retain_old_scores"):
        for strategy in STRATEGIES:
            if strategy == reference:
                continue
            overlap_rows.append(
                {
                    "reference_strategy": reference,
                    "strategy": strategy,
                    **queue_overlap(strategy_masks[reference], strategy_masks[strategy]),
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(out_dir / "strategy_queue_overlap.csv", index=False)

    review_rows = []
    for strategy in STRATEGIES:
        selected = strategy_masks[strategy]
        for regime in sorted(bundles):
            review_rows.append(
                {
                    "strategy": strategy,
                    "review_regime": int(regime),
                    "reviews": int((selected & (strategy_review_regime[strategy] == regime)).sum()),
                    "fraud_reviews": int(
                        (
                            selected
                            & (strategy_review_regime[strategy] == regime)
                            & (future.is_fraud.to_numpy() == 1)
                        ).sum()
                    ),
                }
            )
    pd.DataFrame(review_rows).to_csv(out_dir / "reviews_by_regime.csv", index=False)

    entitlement = int(np.floor(REFERENCE_CAPACITY * len(future) / 10_000))
    if not (metrics.alerts == entitlement).all():
        raise AssertionError("Every strategy must consume the same continuous final capacity")

    profile_rows = {
        int(cycle_id): {
            "alpha": float(_profile(bundle).alpha),
            "amount_scale": float(_profile(bundle).amount_scale),
        }
        for cycle_id, bundle in bundles.items()
    }
    metric_summary = {
        str(row.strategy): {
            key: _json_safe(row[key])
            for key in (
                "alerts",
                "precision",
                "fraud_recall",
                "fraud_value_recall",
                "balanced_hmean",
                "review_delay_mean_steps",
                "review_delay_p90_steps",
                "review_delay_max_steps",
                "fraud_review_delay_mean_steps",
                "fraud_review_delay_p90_steps",
            )
        }
        for _, row in metrics.iterrows()
    }

    summary = {
        "audit": audit,
        "profile": PROFILE,
        "reference_capacity_alerts_per_10k": REFERENCE_CAPACITY,
        "continuous_future_step_min": int(future_step.min()),
        "continuous_future_step_max": int(future_step.max()),
        "continuous_future_n": int(len(future)),
        "continuous_final_review_entitlement": entitlement,
        "refresh_steps": [int(cycles[1].test_step_min), int(cycles[2].test_step_min)],
        "selected_balanced_profile_by_regime": profile_rows,
        "strategy_metrics": metric_summary,
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "All strategies use one continuous analyst-capacity entitlement across steps 595-743; capacity is not reset at model-refresh boundaries.",
            "frozen_incumbent uses the cycle-1 model/calibrator/policy throughout; the three handoff strategies switch to the rolling cycle-2 and cycle-3 systems at steps 645 and 695.",
            "retain_old_scores preserves pending-case scores across refresh, rescore_pending applies the newly released model/policy score to all pending cases, and drop_pending expires all unresolved cases at refresh.",
            "No score from a later model regime is used before that regime's release step. New arrivals are scored only by the model active on arrival.",
            "The rolling model lifecycle retains the v1.8 assumption that prior-period labels are available for the next refresh; PaySim does not contain real label-maturity or investigation-completion timestamps.",
            "Pending analyst-review status is therefore an operations abstraction and is not evidence that the labels used upstream would be unavailable in the same way in production.",
            "PaySim is synthetic; transaction amount is not prevented loss and PaySim steps are not production SLA units."
        ],
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
