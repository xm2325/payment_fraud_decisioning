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
from fraud_decisioning.paysim_causal_promotion import paired_causal_backlog_block_bootstrap
from fraud_decisioning.paysim_features import FEATURE_SETS
from fraud_decisioning.paysim_full import (
    _load_split,
    audit_sql,
    connect_duckdb,
    determine_split,
    materialise_features,
    validate_canonical,
)
from fraud_decisioning.paysim_policy_promotion import PromotionGateConfig, promotion_decision
from fraud_decisioning.paysim_promotion_sensitivity import (
    DEFAULT_BLOCK_STEPS,
    sensitivity_summary,
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
PROMOTION_CONFIG = PromotionGateConfig()
PROMOTION_FAMILY_TESTS = 6
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


def _bootstrap_seed(cycle_id: int, profile: str, block_steps: int) -> int:
    seed = PROMOTION_CONFIG.seed + 100 * int(cycle_id) + PROFILE_SEED_OFFSET[profile]
    if int(block_steps) != PROMOTION_CONFIG.block_steps:
        seed += 1000 * int(block_steps)
    return int(seed)


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


def _predict(bundle, frame):
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    return calibrate(bundle["calibrator"], raw)


def _selected_profile(bundle, profile: str):
    rows = bundle["selected"].loc[bundle["selected"].profile == profile]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row for profile {profile}")
    return rows.iloc[0]


def _promotion_row(
    cycle,
    profile: str,
    block_steps: int,
    static_bundle,
    rolling_bundle,
    test,
    static_probability,
    rolling_probability,
    tail_alpha: float,
):
    incumbent_profile = _selected_profile(static_bundle, profile)
    candidate_profile = _selected_profile(rolling_bundle, profile)
    incumbent_score = priority_score(
        static_probability,
        test.amount,
        float(incumbent_profile.alpha),
        float(incumbent_profile.amount_scale),
    )
    candidate_score = priority_score(
        rolling_probability,
        test.amount,
        float(candidate_profile.alpha),
        float(candidate_profile.amount_scale),
    )
    uncertainty = paired_causal_backlog_block_bootstrap(
        test.step,
        test.is_fraud,
        test.amount,
        test.event_key,
        incumbent_score,
        candidate_score,
        alerts_per_10k=PROMOTION_CONFIG.alerts_per_10k,
        block_steps=int(block_steps),
        n_bootstrap=PROMOTION_CONFIG.n_bootstrap,
        tail_alpha=tail_alpha,
        seed=_bootstrap_seed(cycle.cycle, profile, block_steps),
    )
    decision = promotion_decision(
        profile,
        float(incumbent_profile.alpha),
        float(candidate_profile.alpha),
        uncertainty,
        min_primary_gain=PROMOTION_CONFIG.min_primary_gain,
        precision_noninferiority_margin=PROMOTION_CONFIG.precision_noninferiority_margin,
        recall_noninferiority_margin=PROMOTION_CONFIG.recall_noninferiority_margin,
        value_recall_noninferiority_margin=PROMOTION_CONFIG.value_recall_noninferiority_margin,
    )
    intervals = uncertainty["delta_intervals"]
    guards = decision["guardrails"]
    incumbent = uncertainty["incumbent"]
    candidate = uncertainty["candidate"]
    return {
        "cycle": int(cycle.cycle),
        "profile": profile,
        "routing_contract": "seen_so_far_backlog",
        "block_steps": int(block_steps),
        "bootstrap_seed": _bootstrap_seed(cycle.cycle, profile, block_steps),
        "test_step_min": int(cycle.test_step_min),
        "test_step_max": int(cycle.test_step_max),
        "incumbent_alpha": float(incumbent_profile.alpha),
        "candidate_alpha": float(candidate_profile.alpha),
        "decision": decision["decision"],
        "primary_metric": decision["primary_metric"],
        "primary_point_delta": float(decision["primary_point_delta"]),
        "primary_lower_bound": float(decision["primary_lower_bound"]),
        "incumbent_precision": float(incumbent["precision"]),
        "candidate_precision": float(candidate["precision"]),
        "delta_precision": float(intervals["precision"]["point_delta"]),
        "lcb_precision": float(intervals["precision"]["lower"]),
        "ucb_precision": float(intervals["precision"]["upper"]),
        "incumbent_fraud_recall": float(incumbent["fraud_recall"]),
        "candidate_fraud_recall": float(candidate["fraud_recall"]),
        "delta_fraud_recall": float(intervals["fraud_recall"]["point_delta"]),
        "lcb_fraud_recall": float(intervals["fraud_recall"]["lower"]),
        "ucb_fraud_recall": float(intervals["fraud_recall"]["upper"]),
        "incumbent_fraud_value_recall": float(incumbent["fraud_value_recall"]),
        "candidate_fraud_value_recall": float(candidate["fraud_value_recall"]),
        "delta_fraud_value_recall": float(intervals["fraud_value_recall"]["point_delta"]),
        "lcb_fraud_value_recall": float(intervals["fraud_value_recall"]["lower"]),
        "ucb_fraud_value_recall": float(intervals["fraud_value_recall"]["upper"]),
        "delta_balanced_hmean": float(intervals["balanced_hmean"]["point_delta"]),
        "lcb_balanced_hmean": float(intervals["balanced_hmean"]["lower"]),
        "ucb_balanced_hmean": float(intervals["balanced_hmean"]["upper"]),
        "precision_guardrail_pass": bool(guards.get("precision", True)),
        "fraud_recall_guardrail_pass": bool(guards.get("fraud_recall", True)),
        "fraud_value_recall_guardrail_pass": bool(guards.get("fraud_value_recall", True)),
        "queue_overlap_rate": float(uncertainty["queue_overlap_rate"]),
        "replacement_count": int(uncertainty["replacement_count"]),
        "incumbent_review_delay_mean_steps": float(incumbent["review_delay_mean_steps"]),
        "candidate_review_delay_mean_steps": float(candidate["review_delay_mean_steps"]),
        "incumbent_review_delay_p90_steps": float(incumbent["review_delay_p90_steps"]),
        "candidate_review_delay_p90_steps": float(candidate["review_delay_p90_steps"]),
        "bootstrap_valid": int(uncertainty["n_bootstrap_valid"]),
        "reason": decision["reason"],
    }


def _retrospective_comparison(causal_5: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    reference = reference.loc[reference.cycle.isin([2, 3])].copy()
    columns = [
        "cycle",
        "profile",
        "decision",
        "primary_metric",
        "primary_point_delta",
        "primary_lower_bound",
        "lcb_precision",
        "lcb_fraud_recall",
        "lcb_fraud_value_recall",
    ]
    reference = reference[columns].rename(
        columns={
            column: f"retrospective_{column}"
            for column in columns
            if column not in {"cycle", "profile"}
        }
    )
    causal = causal_5[
        [
            "cycle",
            "profile",
            "decision",
            "primary_metric",
            "primary_point_delta",
            "primary_lower_bound",
            "lcb_precision",
            "lcb_fraud_recall",
            "lcb_fraud_value_recall",
            "queue_overlap_rate",
            "candidate_review_delay_mean_steps",
            "candidate_review_delay_p90_steps",
        ]
    ].rename(
        columns={
            "decision": "causal_decision",
            "primary_metric": "causal_primary_metric",
            "primary_point_delta": "causal_primary_point_delta",
            "primary_lower_bound": "causal_primary_lower_bound",
            "lcb_precision": "causal_lcb_precision",
            "lcb_fraud_recall": "causal_lcb_fraud_recall",
            "lcb_fraud_value_recall": "causal_lcb_fraud_value_recall",
        }
    )
    merged = reference.merge(causal, on=["cycle", "profile"], how="inner", validate="one_to_one")
    if len(merged) != PROMOTION_FAMILY_TESTS:
        raise ValueError("Expected six retrospective-versus-causal promotion comparisons")
    merged["decision_changed"] = merged.retrospective_decision != merged.causal_decision
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_causal_promotion")
    parser.add_argument(
        "--retrospective-reference",
        default="results/paysim_rolling_refresh/policy_promotion_gate.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_causal_promotion_features.parquet"
    db = out_dir / "paysim_causal_promotion.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
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
        raise ValueError("Canonical causal promotion audit expects exactly three rolling cycles")

    first = cycles[0]
    static_train = _load_split(con, work, _where(first.train_step_min, first.train_step_max))
    static_calibration = _load_split(
        con, work, _where(first.calibration_step_min, first.calibration_step_max)
    )
    static_policy = _load_split(con, work, _where(first.policy_step_min, first.policy_step_max))
    static_bundle = _fit_cycle_bundle(static_train, static_calibration, static_policy)
    del static_train, static_calibration, static_policy

    tail_alpha = PROMOTION_CONFIG.family_alpha / PROMOTION_FAMILY_TESTS
    rows: list[dict] = []
    selected_rows: list[pd.DataFrame] = []

    for cycle in cycles[1:]:
        train = _load_split(con, work, _where(cycle.train_step_min, cycle.train_step_max))
        calibration = _load_split(
            con, work, _where(cycle.calibration_step_min, cycle.calibration_step_max)
        )
        policy = _load_split(con, work, _where(cycle.policy_step_min, cycle.policy_step_max))
        rolling_bundle = _fit_cycle_bundle(train, calibration, policy)
        del train, calibration, policy

        selected = rolling_bundle["selected"].copy()
        selected.insert(0, "cycle", int(cycle.cycle))
        selected_rows.append(selected)

        test = _load_split(con, work, _where(cycle.test_step_min, cycle.test_step_max))
        static_probability = _predict(static_bundle, test)
        rolling_probability = _predict(rolling_bundle, test)
        for block_steps in DEFAULT_BLOCK_STEPS:
            for profile in ("case_first", "balanced", "value_first"):
                rows.append(
                    _promotion_row(
                        cycle,
                        profile,
                        int(block_steps),
                        static_bundle,
                        rolling_bundle,
                        test,
                        static_probability,
                        rolling_probability,
                        tail_alpha,
                    )
                )
        del test, static_probability, rolling_probability, rolling_bundle

    block_frame = pd.DataFrame(rows).sort_values(["cycle", "profile", "block_steps"])
    block_frame.to_csv(out_dir / "causal_block_sensitivity.csv", index=False)
    causal_5 = block_frame.loc[block_frame.block_steps == PROMOTION_CONFIG.block_steps].copy()
    causal_5.to_csv(out_dir / "causal_promotion_gate.csv", index=False)

    sensitivity = pd.DataFrame(sensitivity_summary(block_frame.to_dict("records")))
    sensitivity.to_csv(out_dir / "causal_sensitivity_summary.csv", index=False)
    selected_frame = pd.concat(selected_rows, ignore_index=True)
    selected_frame.to_csv(out_dir / "selected_profiles_by_cycle.csv", index=False)

    reference_path = Path(args.retrospective_reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing retrospective reference: {reference_path}")
    comparison = _retrospective_comparison(causal_5, reference_path)
    comparison.to_csv(out_dir / "retrospective_vs_causal_promotion.csv", index=False)

    summary = {
        "audit": audit,
        "routing_contract": "seen_so_far_backlog",
        "reference_capacity_alerts_per_10k": REFERENCE_CAPACITY,
        "block_steps": list(DEFAULT_BLOCK_STEPS),
        "official_gate_block_steps": PROMOTION_CONFIG.block_steps,
        "official_gate_seed_rule": "v1.9 seed + 100*cycle + profile offset",
        "sensitivity_seed_rule": "official seed plus 1000*block_steps for non-5-step runs",
        "bootstrap_replicates_per_comparison": PROMOTION_CONFIG.n_bootstrap,
        "family_tests_per_block_length": PROMOTION_FAMILY_TESTS,
        "family_alpha": PROMOTION_CONFIG.family_alpha,
        "bonferroni_one_sided_tail_alpha": tail_alpha,
        "minimum_primary_gain": PROMOTION_CONFIG.min_primary_gain,
        "noninferiority_margins": {
            "precision": PROMOTION_CONFIG.precision_noninferiority_margin,
            "fraud_recall": PROMOTION_CONFIG.recall_noninferiority_margin,
            "fraud_value_recall": PROMOTION_CONFIG.value_recall_noninferiority_margin,
        },
        "causal_5_step_decisions": causal_5[
            [
                "cycle",
                "profile",
                "decision",
                "primary_metric",
                "primary_point_delta",
                "primary_lower_bound",
            ]
        ].to_dict("records"),
        "dependence_sensitivity": sensitivity.to_dict("records"),
        "retrospective_decision_changes": comparison.loc[comparison.decision_changed].to_dict("records"),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "v1.12 changes the evaluation queue contract, not the pre-declared +2 pp gain threshold, -2 pp guardrails, family-wise alpha or 5-step official block length.",
            "The official 5-step bootstrap uses the same deterministic seed rule as v1.9 so the queue contract is the intended methodological change.",
            "The incumbent and candidate queues use only transactions observed up to each PaySim step; later-step scores cannot affect earlier review decisions.",
            "The paired block intervals resample realised per-step outcomes conditional on the observed frozen causal queues. They do not refit models or reconstruct alternative queues inside each bootstrap replicate.",
            "The 1/3/5/10-step set is a dependence sensitivity audit; the official gate remains the pre-declared 5-step rule.",
            "Completed policy-selection windows may use their full historical scores to choose alpha because those windows precede the test period; future test scores and labels never select alpha.",
            "PaySim is synthetic and its step unit is not presented as a production SLA or investigation timestamp."
        ],
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
