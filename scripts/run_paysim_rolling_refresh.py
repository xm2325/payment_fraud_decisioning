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
from fraud_decisioning.paysim_policy_promotion import (
    PromotionGateConfig,
    paired_circular_block_bootstrap,
    promotion_decision,
)
from fraud_decisioning.paysim_rolling_refresh import (
    build_rolling_cycles,
    cycle_contract_frame,
    validate_cycle_sequence,
)
from fraud_decisioning.paysim_routing_profiles import (
    DEFAULT_ALPHA_GRID,
    evaluate_selected_profiles,
    priority_score,
)
from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)
from fraud_decisioning.paysim_stage_separation import (
    probability_stage_metrics,
    split_validation_stages,
)


REFERENCE_CAPACITY = 50
TEST_WINDOW_STEPS = 50
POLICY_WINDOWS = 3
PROMOTION_CONFIG = PromotionGateConfig()
PROMOTION_FAMILY_TESTS = 6  # cycles 2-3 x three pre-declared routing profiles
PROFILE_SEED_OFFSET = {"case_first": 1, "balanced": 2, "value_first": 3}


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
    calibration_probability = calibrate(calibrator, calibration_raw)
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
    policy_robustness = robustness_summary(policy_grid)
    selected = select_robust_profiles(policy_robustness)
    selected["selection_split"] = "cycle_policy_windows_only"

    return {
        "model": model,
        "calibrator": calibrator,
        "selected": selected,
        "policy_robustness": policy_robustness,
        "calibration_metrics": probability_stage_metrics(
            "calibration", calibration.is_fraud, calibration_probability
        ),
        "policy_metrics": probability_stage_metrics(
            "policy_selection", policy.is_fraud, policy_probability
        ),
    }


def _predict(bundle, frame):
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    return calibrate(bundle["calibrator"], raw)


def _frontier(bundle, frame, probability):
    return evaluate_selected_profiles(
        bundle["selected"],
        frame.is_fraud,
        probability,
        frame.amount,
        frame.event_key,
        budgets_per_10k=(10, 25, 50, 100),
    )


def _comparison_rows(cycle_id: int, static_frontier: pd.DataFrame, rolling_frontier: pd.DataFrame):
    rows = []
    static_50 = static_frontier.loc[
        static_frontier.target_alerts_per_10k == REFERENCE_CAPACITY
    ]
    rolling_50 = rolling_frontier.loc[
        rolling_frontier.target_alerts_per_10k == REFERENCE_CAPACITY
    ]
    for profile in sorted(set(static_50.profile).intersection(set(rolling_50.profile))):
        s = static_50.loc[static_50.profile == profile].iloc[0]
        r = rolling_50.loc[rolling_50.profile == profile].iloc[0]
        rows.append(
            {
                "cycle": int(cycle_id),
                "profile": str(profile),
                "static_alpha": float(s.alpha),
                "rolling_alpha": float(r.alpha),
                "static_precision": float(s.precision),
                "rolling_precision": float(r.precision),
                "delta_precision": float(r.precision - s.precision),
                "static_fraud_recall": float(s.recall),
                "rolling_fraud_recall": float(r.recall),
                "delta_fraud_recall": float(r.recall - s.recall),
                "static_fraud_value_recall": float(s.fraud_value_recall),
                "rolling_fraud_value_recall": float(r.fraud_value_recall),
                "delta_fraud_value_recall": float(
                    r.fraud_value_recall - s.fraud_value_recall
                ),
            }
        )
    return rows


def _selected_profile(bundle, profile: str):
    rows = bundle["selected"].loc[bundle["selected"].profile == profile]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row for profile {profile}")
    return rows.iloc[0]


def _promotion_row(
    cycle,
    profile: str,
    static_bundle,
    rolling_bundle,
    test,
    static_probability,
    rolling_probability,
    tail_alpha: float,
):
    static_profile = _selected_profile(static_bundle, profile)
    rolling_profile = _selected_profile(rolling_bundle, profile)

    if cycle.cycle == 1:
        decision = promotion_decision(
            profile,
            float(static_profile.alpha),
            float(rolling_profile.alpha),
            uncertainty=None,
            policy_changed=False,
        )
        return {
            "cycle": cycle.cycle,
            "profile": profile,
            "test_step_min": cycle.test_step_min,
            "test_step_max": cycle.test_step_max,
            "incumbent_alpha": float(static_profile.alpha),
            "candidate_alpha": float(rolling_profile.alpha),
            "decision": decision["decision"],
            "primary_metric": decision["primary_metric"],
            "primary_point_delta": math.nan,
            "primary_lower_bound": math.nan,
            "delta_precision": math.nan,
            "lcb_precision": math.nan,
            "ucb_precision": math.nan,
            "delta_fraud_recall": math.nan,
            "lcb_fraud_recall": math.nan,
            "ucb_fraud_recall": math.nan,
            "delta_fraud_value_recall": math.nan,
            "lcb_fraud_value_recall": math.nan,
            "ucb_fraud_value_recall": math.nan,
            "delta_balanced_hmean": math.nan,
            "lcb_balanced_hmean": math.nan,
            "ucb_balanced_hmean": math.nan,
            "precision_guardrail_pass": True,
            "fraud_recall_guardrail_pass": True,
            "fraud_value_recall_guardrail_pass": True,
            "bootstrap_valid": 0,
            "reason": decision["reason"],
        }

    static_score = priority_score(
        static_probability,
        test.amount,
        float(static_profile.alpha),
        float(static_profile.amount_scale),
    )
    rolling_score = priority_score(
        rolling_probability,
        test.amount,
        float(rolling_profile.alpha),
        float(rolling_profile.amount_scale),
    )
    uncertainty = paired_circular_block_bootstrap(
        test.step,
        test.is_fraud,
        test.amount,
        test.event_key,
        static_score,
        rolling_score,
        alerts_per_10k=PROMOTION_CONFIG.alerts_per_10k,
        block_steps=PROMOTION_CONFIG.block_steps,
        n_bootstrap=PROMOTION_CONFIG.n_bootstrap,
        tail_alpha=tail_alpha,
        seed=PROMOTION_CONFIG.seed + 100 * cycle.cycle + PROFILE_SEED_OFFSET[profile],
    )
    decision = promotion_decision(
        profile,
        float(static_profile.alpha),
        float(rolling_profile.alpha),
        uncertainty,
        min_primary_gain=PROMOTION_CONFIG.min_primary_gain,
        precision_noninferiority_margin=PROMOTION_CONFIG.precision_noninferiority_margin,
        recall_noninferiority_margin=PROMOTION_CONFIG.recall_noninferiority_margin,
        value_recall_noninferiority_margin=PROMOTION_CONFIG.value_recall_noninferiority_margin,
    )
    intervals = uncertainty["delta_intervals"]
    guards = decision["guardrails"]
    return {
        "cycle": cycle.cycle,
        "profile": profile,
        "test_step_min": cycle.test_step_min,
        "test_step_max": cycle.test_step_max,
        "incumbent_alpha": float(static_profile.alpha),
        "candidate_alpha": float(rolling_profile.alpha),
        "decision": decision["decision"],
        "primary_metric": decision["primary_metric"],
        "primary_point_delta": float(decision["primary_point_delta"]),
        "primary_lower_bound": float(decision["primary_lower_bound"]),
        "delta_precision": float(intervals["precision"]["point_delta"]),
        "lcb_precision": float(intervals["precision"]["lower"]),
        "ucb_precision": float(intervals["precision"]["upper"]),
        "delta_fraud_recall": float(intervals["fraud_recall"]["point_delta"]),
        "lcb_fraud_recall": float(intervals["fraud_recall"]["lower"]),
        "ucb_fraud_recall": float(intervals["fraud_recall"]["upper"]),
        "delta_fraud_value_recall": float(intervals["fraud_value_recall"]["point_delta"]),
        "lcb_fraud_value_recall": float(intervals["fraud_value_recall"]["lower"]),
        "ucb_fraud_value_recall": float(intervals["fraud_value_recall"]["upper"]),
        "delta_balanced_hmean": float(intervals["balanced_hmean"]["point_delta"]),
        "lcb_balanced_hmean": float(intervals["balanced_hmean"]["lower"]),
        "ucb_balanced_hmean": float(intervals["balanced_hmean"]["upper"]),
        "precision_guardrail_pass": bool(guards.get("precision", True)),
        "fraud_recall_guardrail_pass": bool(guards.get("fraud_recall", True)),
        "fraud_value_recall_guardrail_pass": bool(guards.get("fraud_value_recall", True)),
        "bootstrap_valid": int(uncertainty["n_bootstrap_valid"]),
        "reason": decision["reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_rolling_refresh")
    parser.add_argument("--test-window-steps", type=int, default=TEST_WINDOW_STEPS)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_rolling_refresh_features.parquet"
    db = out_dir / "paysim_rolling_refresh.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {
        k: (float(v) if k == "fraud_rate" else int(v))
        for k, v in audit_row.items()
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
        test_window_n_steps=args.test_window_steps,
    )
    validate_cycle_sequence(cycles)
    contract = cycle_contract_frame(cycles)
    contract.to_csv(out_dir / "cycle_contract.csv", index=False)
    del validation

    if len(cycles) != 3:
        raise ValueError("Canonical PaySim v1.9 promotion family expects exactly three rolling cycles")

    first = cycles[0]
    static_train = _load_split(con, work, _where(first.train_step_min, first.train_step_max))
    static_calibration = _load_split(
        con, work, _where(first.calibration_step_min, first.calibration_step_max)
    )
    static_policy = _load_split(
        con, work, _where(first.policy_step_min, first.policy_step_max)
    )
    static_bundle = _fit_cycle_bundle(static_train, static_calibration, static_policy)
    del static_train, static_calibration, static_policy

    frontier_frames = []
    probability_rows = []
    selected_frames = []
    robustness_frames = []
    comparison_rows = []
    promotion_rows = []
    promotion_tail_alpha = PROMOTION_CONFIG.family_alpha / PROMOTION_FAMILY_TESTS

    for cycle in cycles:
        if cycle.cycle == 1:
            rolling_bundle = static_bundle
        else:
            train = _load_split(
                con, work, _where(cycle.train_step_min, cycle.train_step_max)
            )
            calibration = _load_split(
                con,
                work,
                _where(cycle.calibration_step_min, cycle.calibration_step_max),
            )
            policy = _load_split(
                con, work, _where(cycle.policy_step_min, cycle.policy_step_max)
            )
            rolling_bundle = _fit_cycle_bundle(train, calibration, policy)
            del train, calibration, policy

        selected = rolling_bundle["selected"].copy()
        selected.insert(0, "cycle", cycle.cycle)
        selected.insert(1, "test_step_min", cycle.test_step_min)
        selected.insert(2, "test_step_max", cycle.test_step_max)
        selected_frames.append(selected)

        robust = rolling_bundle["policy_robustness"].copy()
        robust.insert(0, "cycle", cycle.cycle)
        robust.insert(1, "policy_step_min", cycle.policy_step_min)
        robust.insert(2, "policy_step_max", cycle.policy_step_max)
        robustness_frames.append(robust)

        for stage_name, metrics in (
            ("rolling_calibration", rolling_bundle["calibration_metrics"]),
            ("rolling_policy_selection", rolling_bundle["policy_metrics"]),
        ):
            row = dict(metrics)
            row.update(
                {
                    "cycle": cycle.cycle,
                    "strategy": stage_name,
                    "test_step_min": cycle.test_step_min,
                    "test_step_max": cycle.test_step_max,
                }
            )
            probability_rows.append(row)

        test = _load_split(con, work, _where(cycle.test_step_min, cycle.test_step_max))
        static_probability = _predict(static_bundle, test)
        rolling_probability = _predict(rolling_bundle, test)

        for strategy, probability in (
            ("frozen_v1_7", static_probability),
            ("rolling_refresh", rolling_probability),
        ):
            row = probability_stage_metrics(
                f"cycle_{cycle.cycle}_test", test.is_fraud, probability
            )
            row.update(
                {
                    "cycle": cycle.cycle,
                    "strategy": strategy,
                    "test_step_min": cycle.test_step_min,
                    "test_step_max": cycle.test_step_max,
                }
            )
            probability_rows.append(row)

        static_frontier = _frontier(static_bundle, test, static_probability)
        static_frontier.insert(0, "cycle", cycle.cycle)
        static_frontier.insert(1, "strategy", "frozen_v1_7")
        static_frontier.insert(2, "test_step_min", cycle.test_step_min)
        static_frontier.insert(3, "test_step_max", cycle.test_step_max)

        rolling_frontier = _frontier(rolling_bundle, test, rolling_probability)
        rolling_frontier.insert(0, "cycle", cycle.cycle)
        rolling_frontier.insert(1, "strategy", "rolling_refresh")
        rolling_frontier.insert(2, "test_step_min", cycle.test_step_min)
        rolling_frontier.insert(3, "test_step_max", cycle.test_step_max)

        frontier_frames.extend([static_frontier, rolling_frontier])
        comparison_rows.extend(
            _comparison_rows(cycle.cycle, static_frontier, rolling_frontier)
        )

        for profile in ("case_first", "balanced", "value_first"):
            promotion_rows.append(
                _promotion_row(
                    cycle,
                    profile,
                    static_bundle,
                    rolling_bundle,
                    test,
                    static_probability,
                    rolling_probability,
                    promotion_tail_alpha,
                )
            )
        del test

    frontiers = pd.concat(frontier_frames, ignore_index=True)
    frontiers.to_csv(out_dir / "routing_frontier_by_cycle.csv", index=False)
    probability_frame = pd.DataFrame(probability_rows)
    probability_frame.to_csv(out_dir / "probability_diagnostics_by_cycle.csv", index=False)
    selected_frame = pd.concat(selected_frames, ignore_index=True)
    selected_frame.to_csv(out_dir / "selected_profiles_by_cycle.csv", index=False)
    robustness_frame = pd.concat(robustness_frames, ignore_index=True)
    robustness_frame.to_csv(out_dir / "policy_robustness_by_cycle.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "routing_comparison_50_per_10k.csv", index=False)
    promotion = pd.DataFrame(promotion_rows)
    promotion.to_csv(out_dir / "policy_promotion_gate.csv", index=False)

    balanced = comparison.loc[comparison.profile == "balanced"].copy()
    tested_promotions = promotion.loc[promotion.cycle > 1]
    summary = {
        "audit": audit,
        "outer_split": {
            "train_cut": outer_split.train_cut,
            "future_cut": outer_split.validation_cut,
        },
        "base_stage_lengths": {
            "calibration_n_steps": stage_split.calibration_n_steps,
            "policy_n_steps": stage_split.policy_n_steps,
            "test_window_n_steps": int(args.test_window_steps),
        },
        "n_cycles": len(cycles),
        "cycle_contract": contract.to_dict(orient="records"),
        "model": "transaction_plus_relational",
        "reference_capacity_alerts_per_10k": REFERENCE_CAPACITY,
        "alpha_grid": list(DEFAULT_ALPHA_GRID),
        "balanced_50_per_10k_comparison": balanced.to_dict(orient="records"),
        "promotion_gate": {
            "family_tests": PROMOTION_FAMILY_TESTS,
            "family_alpha": PROMOTION_CONFIG.family_alpha,
            "bonferroni_one_sided_tail_alpha": promotion_tail_alpha,
            "block_steps": PROMOTION_CONFIG.block_steps,
            "n_bootstrap": PROMOTION_CONFIG.n_bootstrap,
            "min_primary_gain": PROMOTION_CONFIG.min_primary_gain,
            "precision_noninferiority_margin": PROMOTION_CONFIG.precision_noninferiority_margin,
            "recall_noninferiority_margin": PROMOTION_CONFIG.recall_noninferiority_margin,
            "value_recall_noninferiority_margin": PROMOTION_CONFIG.value_recall_noninferiority_margin,
            "decisions": tested_promotions[
                [
                    "cycle",
                    "profile",
                    "incumbent_alpha",
                    "candidate_alpha",
                    "decision",
                    "primary_metric",
                    "primary_point_delta",
                    "primary_lower_bound",
                ]
            ].to_dict(orient="records"),
        },
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "Cycle 1 exactly preserves the v1.7 temporal stages: train 1-445, calibration 446-519, policy selection 520-594, then test from step 595.",
            "Later cycles move the origin forward and may use labels from earlier completed test windows, but never labels from their own or later test windows.",
            "The frozen_v1_7 comparator never retrains, recalibrates or reselects routing alpha after cycle 1.",
            "The rolling_refresh strategy refits the model on expanding history, refits calibration on the declared calibration window and reselects alpha only on the immediately preceding policy window.",
            "v1.9 treats each refreshed cycle/profile score as a candidate policy even when alpha is unchanged; the model, calibrator, amount scale or queue ordering may still differ.",
            "Promotion evidence uses a paired circular 5-step block bootstrap on a completed test window with Bonferroni one-sided family control across the six cycle-2/3 profile comparisons.",
            "A candidate is not promoted unless its pre-declared primary metric gains at least 2 percentage points, its family-adjusted lower bound is positive, and non-primary guardrails are non-inferior within declared margins.",
            "A promotion decision is retrospective after that test window has mature labels; it cannot alter or claim performance on the same test window.",
            "The block bootstrap measures uncertainty in incremental captured outcomes for frozen queues; it does not include model-fit uncertainty or guarantee future production performance.",
            "PaySim does not contain real fraud-label maturity or investigation delay. This rolling refresh is therefore an as-of upper-bound under labels being available by the next refresh, not delayed-label production evidence.",
            "The controlled 120k stress-test layer remains the evidence source for explicit delayed-label and verification-bias behaviour.",
            "PaySim is synthetic mobile-money data; results are benchmark evidence rather than production impact, prevented loss or staffing estimates.",
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
