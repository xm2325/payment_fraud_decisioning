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
from fraud_decisioning.paysim_promotion_sensitivity import (
    DEFAULT_BLOCK_STEPS,
    sensitivity_summary,
)
from fraud_decisioning.paysim_rolling_refresh import (
    build_rolling_cycles,
    validate_cycle_sequence,
)
from fraud_decisioning.paysim_routing_profiles import (
    DEFAULT_ALPHA_GRID,
    priority_score,
)
from fraud_decisioning.paysim_routing_robustness import (
    robustness_summary,
    select_robust_profiles,
    validation_window_alpha_grid,
)
from fraud_decisioning.paysim_stage_separation import split_validation_stages


REFERENCE_CAPACITY = 50
TEST_WINDOW_STEPS = 50
POLICY_WINDOWS = 3
FAMILY_TESTS = 6
CONFIG = PromotionGateConfig()
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


def _fit_bundle(train, calibration, policy):
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
    return {"model": model, "calibrator": calibrator, "selected": selected}


def _predict(bundle, frame):
    features = FEATURE_SETS["transaction_plus_relational"]
    raw = bundle["model"].predict_proba(frame[features])[:, 1]
    return calibrate(bundle["calibrator"], raw)


def _profile(bundle, profile: str):
    rows = bundle["selected"].loc[bundle["selected"].profile == profile]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row for profile {profile}")
    return rows.iloc[0]


def _sensitivity_row(
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
    incumbent = _profile(static_bundle, profile)
    candidate = _profile(rolling_bundle, profile)
    incumbent_score = priority_score(
        static_probability,
        test.amount,
        float(incumbent.alpha),
        float(incumbent.amount_scale),
    )
    candidate_score = priority_score(
        rolling_probability,
        test.amount,
        float(candidate.alpha),
        float(candidate.amount_scale),
    )
    uncertainty = paired_circular_block_bootstrap(
        test.step,
        test.is_fraud,
        test.amount,
        test.event_key,
        incumbent_score,
        candidate_score,
        alerts_per_10k=CONFIG.alerts_per_10k,
        block_steps=block_steps,
        n_bootstrap=CONFIG.n_bootstrap,
        tail_alpha=tail_alpha,
        seed=CONFIG.seed + 100 * cycle.cycle + PROFILE_SEED_OFFSET[profile],
    )
    decision = promotion_decision(
        profile,
        float(incumbent.alpha),
        float(candidate.alpha),
        uncertainty,
        min_primary_gain=CONFIG.min_primary_gain,
        precision_noninferiority_margin=CONFIG.precision_noninferiority_margin,
        recall_noninferiority_margin=CONFIG.recall_noninferiority_margin,
        value_recall_noninferiority_margin=CONFIG.value_recall_noninferiority_margin,
    )
    intervals = uncertainty["delta_intervals"]
    guards = decision["guardrails"]
    return {
        "cycle": int(cycle.cycle),
        "profile": profile,
        "block_steps": int(block_steps),
        "incumbent_alpha": float(incumbent.alpha),
        "candidate_alpha": float(candidate.alpha),
        "decision": decision["decision"],
        "primary_metric": decision["primary_metric"],
        "primary_point_delta": float(decision["primary_point_delta"]),
        "primary_lower_bound": float(decision["primary_lower_bound"]),
        "delta_precision": float(intervals["precision"]["point_delta"]),
        "lcb_precision": float(intervals["precision"]["lower"]),
        "delta_fraud_recall": float(intervals["fraud_recall"]["point_delta"]),
        "lcb_fraud_recall": float(intervals["fraud_recall"]["lower"]),
        "delta_fraud_value_recall": float(intervals["fraud_value_recall"]["point_delta"]),
        "lcb_fraud_value_recall": float(intervals["fraud_value_recall"]["lower"]),
        "delta_balanced_hmean": float(intervals["balanced_hmean"]["point_delta"]),
        "lcb_balanced_hmean": float(intervals["balanced_hmean"]["lower"]),
        "precision_guardrail_pass": bool(guards.get("precision", True)),
        "fraud_recall_guardrail_pass": bool(guards.get("fraud_recall", True)),
        "fraud_value_recall_guardrail_pass": bool(guards.get("fraud_value_recall", True)),
        "bootstrap_valid": int(uncertainty["n_bootstrap_valid"]),
    }


def _validate_v19_reference(sensitivity: pd.DataFrame, reference_path: Path) -> None:
    reference = pd.read_csv(reference_path)
    reference = reference.loc[reference.cycle > 1].copy()
    block5 = sensitivity.loc[sensitivity.block_steps == CONFIG.block_steps].copy()
    key = ["cycle", "profile"]
    merged = block5.merge(reference, on=key, suffixes=("_v110", "_v19"), validate="one_to_one")
    if len(merged) != FAMILY_TESTS:
        raise ValueError("v1.9 reference comparison must contain six cycle/profile rows")
    for col in (
        "primary_point_delta",
        "primary_lower_bound",
        "lcb_precision",
        "lcb_fraud_recall",
        "lcb_fraud_value_recall",
    ):
        left = merged[f"{col}_v110"].to_numpy(dtype=float)
        right = merged[f"{col}_v19"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"5-step sensitivity no longer reproduces v1.9 column {col}")
    if not np.array_equal(
        merged["decision_v110"].to_numpy(), merged["decision_v19"].to_numpy()
    ):
        raise AssertionError("5-step sensitivity no longer reproduces v1.9 decisions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--out", default="results/paysim_promotion_sensitivity")
    parser.add_argument(
        "--v1-9-reference",
        default="results/paysim_rolling_refresh/policy_promotion_gate.csv",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "features.parquet"
    db = out_dir / "audit.duckdb"
    con = connect_duckdb(db)
    started = time.time()

    audit_row = con.execute(audit_sql(args.parquet_glob)).df().iloc[0].to_dict()
    audit = {
        k: (float(v) if k == "fraud_rate" else int(v))
        for k, v in audit_row.items()
    }
    validate_canonical(audit)
    outer = determine_split(con, args.parquet_glob)
    materialise_features(con, args.parquet_glob, work)

    validation = _load_split(
        con, work, f"step >= {outer.train_cut} AND step < {outer.validation_cut}"
    )
    stage = split_validation_stages(validation.step, calibration_fraction=0.5)
    del validation
    work_sql = str(work).replace("'", "''")
    all_steps = con.execute(
        f"SELECT DISTINCT step::INTEGER AS step FROM read_parquet('{work_sql}') ORDER BY step"
    ).fetchnumpy()["step"]
    cycles = build_rolling_cycles(
        all_steps,
        initial_test_step=outer.validation_cut,
        calibration_n_steps=stage.calibration_n_steps,
        policy_n_steps=stage.policy_n_steps,
        test_window_n_steps=TEST_WINDOW_STEPS,
    )
    validate_cycle_sequence(cycles)
    if len(cycles) != 3:
        raise ValueError("Canonical PaySim sensitivity audit expects three cycles")

    first = cycles[0]
    train = _load_split(con, work, _where(first.train_step_min, first.train_step_max))
    calibration = _load_split(
        con, work, _where(first.calibration_step_min, first.calibration_step_max)
    )
    policy = _load_split(con, work, _where(first.policy_step_min, first.policy_step_max))
    static_bundle = _fit_bundle(train, calibration, policy)
    del train, calibration, policy

    tail_alpha = CONFIG.family_alpha / FAMILY_TESTS
    rows = []
    for cycle in cycles[1:]:
        train = _load_split(con, work, _where(cycle.train_step_min, cycle.train_step_max))
        calibration = _load_split(
            con, work, _where(cycle.calibration_step_min, cycle.calibration_step_max)
        )
        policy = _load_split(con, work, _where(cycle.policy_step_min, cycle.policy_step_max))
        rolling_bundle = _fit_bundle(train, calibration, policy)
        del train, calibration, policy

        test = _load_split(con, work, _where(cycle.test_step_min, cycle.test_step_max))
        static_probability = _predict(static_bundle, test)
        rolling_probability = _predict(rolling_bundle, test)
        for profile in ("case_first", "balanced", "value_first"):
            for block_steps in DEFAULT_BLOCK_STEPS:
                rows.append(
                    _sensitivity_row(
                        cycle,
                        profile,
                        block_steps,
                        static_bundle,
                        rolling_bundle,
                        test,
                        static_probability,
                        rolling_probability,
                        tail_alpha,
                    )
                )
        del test

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(out_dir / "block_sensitivity.csv", index=False)
    summary_rows = sensitivity_summary(rows)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    reference_path = Path(args.v1_9_reference)
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing frozen v1.9 reference: {reference_path}")
    _validate_v19_reference(sensitivity, reference_path)

    key = summary_frame.loc[
        (summary_frame.cycle == 3) & (summary_frame.profile == "value_first")
    ]
    if len(key) != 1:
        raise ValueError("Missing cycle-3 value-first sensitivity summary")

    summary = {
        "audit": audit,
        "block_steps": list(DEFAULT_BLOCK_STEPS),
        "bootstrap_replicates_per_comparison": CONFIG.n_bootstrap,
        "family_tests_per_block_length": FAMILY_TESTS,
        "family_alpha": CONFIG.family_alpha,
        "bonferroni_one_sided_tail_alpha": tail_alpha,
        "frozen_v1_9_block_steps": CONFIG.block_steps,
        "v1_9_reference_reproduced": True,
        "cycle_3_value_first": key.iloc[0].to_dict(),
        "all_sensitivity_classes": summary_frame.to_dict(orient="records"),
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundaries": [
            "v1.10 does not change the frozen v1.9 5-step promotion rule or its KEEP_INCUMBENT decision.",
            "Block lengths 1, 3, 5 and 10 are a pre-declared dependence sensitivity set, not alternative settings chosen after seeing which one promotes a candidate.",
            "Each block length reruns the same six cycle-2/3 profile comparisons with the same family-wise alpha and promotion guardrails.",
            "ROBUST_KEEP_INCUMBENT means every declared block length rejects promotion; DEPENDENCE_SENSITIVE means the decision changes with the dependence assumption.",
            "The 5-step rows must exactly reproduce the committed v1.9 policy-promotion result before the sensitivity audit is accepted.",
            "These intervals quantify temporal uncertainty in realised frozen-queue outcomes; they do not include retraining uncertainty or guarantee future production performance.",
            "PaySim is synthetic and lacks real investigation-completion and label-maturity timestamps."
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    work.unlink(missing_ok=True)
    con.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
