from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .paysim_metrics import binary_metrics, rule_metrics, threshold_at_legit_rate

CANONICAL_PAYSIM = {
    "n": 6_362_620,
    "fraud_n": 8_213,
    "step_min": 1,
    "step_max": 743,
}

from .paysim_features import BALANCE_FREE_CANDIDATES, FEATURE_COLUMNS, FEATURE_SETS


@dataclass(frozen=True)
class PaySimSplit:
    train_cut: int
    validation_cut: int


def connect_duckdb(database: str | Path = ":memory:"):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised on GitHub runner
        raise RuntimeError("Full PaySim benchmark requires duckdb>=1.1") from exc
    con = duckdb.connect(str(database))
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")
    return con


def parquet_expr(parquet_glob: str | Path) -> str:
    path = str(parquet_glob).replace("'", "''")
    return f"read_parquet('{path}')"


def audit_sql(parquet_glob: str | Path) -> str:
    src = parquet_expr(parquet_glob)
    return f"""
        SELECT
            COUNT(*)::BIGINT AS n,
            SUM(isFraud)::BIGINT AS fraud_n,
            MIN(step)::INTEGER AS step_min,
            MAX(step)::INTEGER AS step_max,
            AVG(isFraud)::DOUBLE AS fraud_rate
        FROM {src}
    """


def validate_canonical(audit: dict) -> None:
    for key, expected in CANONICAL_PAYSIM.items():
        actual = int(audit[key])
        if actual != expected:
            raise ValueError(f"PaySim canonical audit failed for {key}: {actual} != {expected}")


def determine_split(con, parquet_glob: str | Path) -> PaySimSplit:
    src = parquet_expr(parquet_glob)
    steps = con.execute(f"SELECT DISTINCT step::INTEGER AS step FROM {src} ORDER BY step").fetchnumpy()["step"]
    if len(steps) < 10:
        raise ValueError("Need at least 10 distinct PaySim steps")
    return PaySimSplit(
        train_cut=int(steps[int(len(steps) * 0.60)]),
        validation_cut=int(steps[int(len(steps) * 0.80)]),
    )


def feature_sql(parquet_glob: str | Path) -> str:
    """DuckDB SQL with strict prior-step windows; same-step events cannot see each other."""
    src = parquet_expr(parquet_glob)
    return f"""
    WITH base AS (
        SELECT
            step::INTEGER AS step,
            type::VARCHAR AS type,
            amount::DOUBLE AS amount,
            nameOrig::VARCHAR AS sender,
            nameDest::VARCHAR AS recipient,
            oldbalanceOrg::DOUBLE AS old_balance_sender,
            newbalanceOrig::DOUBLE AS new_balance_sender,
            oldbalanceDest::DOUBLE AS old_balance_dest,
            newbalanceDest::DOUBLE AS new_balance_dest,
            isFraud::INTEGER AS is_fraud,
            isFlaggedFraud::INTEGER AS is_flagged_fraud,
            COUNT(*) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS sender_tx_1h,
            COUNT(*) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS sender_tx_24h,
            COALESCE(SUM(amount) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            ), 0)::DOUBLE AS sender_amount_24h,
            COUNT(*) OVER (
                PARTITION BY nameDest ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS recipient_fanin_24h,
            AVG(amount) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS sender_mean_amount_7d,
            COUNT(*) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS sender_tx_7d,
            COUNT(*) OVER (
                PARTITION BY nameDest ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS recipient_tx_7d,
            COALESCE(SUM(amount) OVER (
                PARTITION BY nameDest ORDER BY step
                RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
            ), 0)::DOUBLE AS recipient_amount_24h,
            COUNT(*) OVER (
                PARTITION BY nameOrig, nameDest ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS pair_tx_7d,
            COALESCE(SUM(amount) OVER (
                PARTITION BY nameOrig, nameDest ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            ), 0)::DOUBLE AS pair_amount_7d,
            COALESCE(SUM(amount) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            ), 0)::DOUBLE AS sender_amount_7d,
            APPROX_COUNT_DISTINCT(nameDest) OVER (
                PARTITION BY nameOrig ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS sender_unique_recipients_7d,
            APPROX_COUNT_DISTINCT(nameOrig) OVER (
                PARTITION BY nameDest ORDER BY step
                RANGE BETWEEN 168 PRECEDING AND 1 PRECEDING
            )::DOUBLE AS recipient_unique_senders_7d
        FROM {src}
    )
    SELECT
        step,
        amount,
        is_fraud,
        is_flagged_fraud,
        LN(1 + GREATEST(amount, 0))::DOUBLE AS log_amount,
        LEAST(GREATEST(amount / GREATEST(old_balance_sender, 1), 0), 20)::DOUBLE AS balance_fraction,
        sender_tx_1h,
        sender_tx_24h,
        sender_amount_24h,
        recipient_fanin_24h,
        (amount / GREATEST(COALESCE(sender_mean_amount_7d, amount), 1))::DOUBLE AS amount_vs_7d_mean,
        sender_tx_7d,
        recipient_tx_7d,
        recipient_amount_24h,
        pair_tx_7d,
        pair_amount_7d,
        LEAST(GREATEST(pair_amount_7d / GREATEST(sender_amount_7d, 1), 0), 1)::DOUBLE AS sender_recipient_share_7d,
        sender_unique_recipients_7d,
        recipient_unique_senders_7d,
        (old_balance_sender - new_balance_sender)::DOUBLE AS orig_balance_delta,
        (new_balance_dest - old_balance_dest)::DOUBLE AS dest_balance_delta,
        (type = 'CASH_IN')::INTEGER AS type_CASH_IN,
        (type = 'CASH_OUT')::INTEGER AS type_CASH_OUT,
        (type = 'DEBIT')::INTEGER AS type_DEBIT,
        (type = 'PAYMENT')::INTEGER AS type_PAYMENT,
        (type = 'TRANSFER')::INTEGER AS type_TRANSFER
    FROM base
    """


def materialise_features(con, parquet_glob: str | Path, output_parquet: str | Path) -> None:
    out = str(output_parquet).replace("'", "''")
    con.execute(f"COPY ({feature_sql(parquet_glob)}) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def _load_split(con, feature_parquet: str | Path, where: str) -> pd.DataFrame:
    p = str(feature_parquet).replace("'", "''")
    cols = ", ".join(["step", "amount", "is_fraud", "is_flagged_fraud"] + FEATURE_COLUMNS)
    return con.execute(f"SELECT {cols} FROM read_parquet('{p}') WHERE {where}").df()


def run_full_benchmark(parquet_glob: str | Path, out_dir: str | Path) -> dict:
    from .modeling import fit_lightgbm, fit_sigmoid_calibrator, calibrate

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "paysim_features.parquet"
    con = connect_duckdb(out_dir / "paysim.duckdb")
    started = time.time()

    audit_row = con.execute(audit_sql(parquet_glob)).df().iloc[0].to_dict()
    audit = {k: (float(v) if k == "fraud_rate" else int(v)) for k, v in audit_row.items()}
    validate_canonical(audit)
    split = determine_split(con, parquet_glob)
    materialise_features(con, parquet_glob, work)

    train = _load_split(con, work, f"step < {split.train_cut}")
    val = _load_split(con, work, f"step >= {split.train_cut} AND step < {split.validation_cut}")
    test = _load_split(con, work, f"step >= {split.validation_cut}")

    predictions = {}
    ablation_rows = []
    for model_name, features in FEATURE_SETS.items():
        model = fit_lightgbm(train[features], train.is_fraud, n_estimators=250)
        pv_raw = model.predict_proba(val[features])[:, 1]
        cal = fit_sigmoid_calibrator(pv_raw, val.is_fraud)
        pv = calibrate(cal, pv_raw)
        pt = calibrate(cal, model.predict_proba(test[features])[:, 1])
        predictions[model_name] = (pv, pt)

        threshold = threshold_at_legit_rate(val.is_fraud.to_numpy(), pv, 0.001)
        row = binary_metrics(test.is_fraud, pt, test.amount, threshold)
        val_row = binary_metrics(val.is_fraud, pv, val.amount, threshold)
        row.update({
            "model": model_name,
            "n_features": len(features),
            "validation_pr_auc": val_row["pr_auc"],
            "target_validation_legit_flag_rate": 0.001,
        })
        ablation_rows.append(row)

    pd.DataFrame(ablation_rows).to_csv(out_dir / "model_ablation.csv", index=False)

    # Select the balance-free champion on validation only; simulator balances remain sensitivity-only.
    balance_free_rows = [r for r in ablation_rows if r["model"] in BALANCE_FREE_CANDIDATES]
    balance_free_reference = max(balance_free_rows, key=lambda r: r["validation_pr_auc"])["model"]
    pv, pt = predictions[balance_free_reference]
    operating_rows = []
    for target in [0.0002, 0.0005, 0.001, 0.002, 0.005]:
        threshold = threshold_at_legit_rate(val.is_fraud.to_numpy(), pv, target)
        row = binary_metrics(test.is_fraud, pt, test.amount, threshold)
        row["target_validation_legit_flag_rate"] = target
        operating_rows.append(row)
    pd.DataFrame(operating_rows).to_csv(out_dir / "operating_points.csv", index=False)

    source_rule = rule_metrics(test.is_fraud, test.is_flagged_fraud, test.amount)

    # Interpretable amount/type rule: threshold is chosen on validation only.
    val_rule_score = np.where(
        (val["type_TRANSFER"].to_numpy() + val["type_CASH_OUT"].to_numpy()) > 0,
        val["log_amount"].to_numpy(),
        -1e9,
    )
    test_rule_score = np.where(
        (test["type_TRANSFER"].to_numpy() + test["type_CASH_OUT"].to_numpy()) > 0,
        test["log_amount"].to_numpy(),
        -1e9,
    )
    simple_rule_threshold = threshold_at_legit_rate(val.is_fraud.to_numpy(), val_rule_score, 0.001)
    simple_rule = binary_metrics(test.is_fraud, test_rule_score, test.amount, simple_rule_threshold)
    simple_rule["threshold_log_amount"] = float(simple_rule_threshold)
    simple_rule["target_validation_legit_flag_rate"] = 0.001

    fraud_amounts = np.sort(test.loc[test.is_fraud == 1, "amount"].to_numpy())[::-1]
    total_fraud_value = float(fraud_amounts.sum())
    value_concentration = {}
    for share in (0.10, 0.25, 0.50):
        k = max(1, int(np.ceil(len(fraud_amounts) * share)))
        value_concentration[f"top_{int(share*100)}pct_cases_value_share"] = float(
            fraud_amounts[:k].sum() / total_fraud_value
        )

    split_rows = [
        {"split": "train", "n": len(train), "fraud_n": int(train.is_fraud.sum()), "fraud_rate": float(train.is_fraud.mean()), "step_min": int(train.step.min()), "step_max": int(train.step.max())},
        {"split": "validation", "n": len(val), "fraud_n": int(val.is_fraud.sum()), "fraud_rate": float(val.is_fraud.mean()), "step_min": int(val.step.min()), "step_max": int(val.step.max())},
        {"split": "test", "n": len(test), "fraud_n": int(test.is_fraud.sum()), "fraud_rate": float(test.is_fraud.mean()), "step_min": int(test.step.min()), "step_max": int(test.step.max())},
    ]
    pd.DataFrame(split_rows).to_csv(out_dir / "split_summary.csv", index=False)

    summary = {
        "audit": audit,
        "split": {"train_cut": split.train_cut, "validation_cut": split.validation_cut},
        "source_rule": source_rule,
        "simple_amount_type_rule": simple_rule,
        "fraud_value_concentration": value_concentration,
        "feature_sets": FEATURE_SETS,
        "model_ablation_reference": {row["model"]: row for row in ablation_rows},
        "runtime_seconds": float(time.time() - started),
        "balance_free_reference_model": balance_free_reference,
        "reference_operating_point": operating_rows[2],
        "limitations": [
            "PaySim is synthetic mobile-money data; performance is not a production estimate.",
            "Balance fields reflect simulator mechanics and are sensitivity-only; the balance-free champion is selected on validation PR-AUC only.",
            "Approximate distinct counterparty counts are strict prior-step 7-day windows and are evaluated only for incremental value.",
            "The simple amount/type rule threshold is also selected on validation before future-test evaluation.",
            "Step is hourly; same-step transactions are treated as simultaneous and cannot see each other.",
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    work.unlink(missing_ok=True)
    con.close()
    (out_dir / "paysim.duckdb").unlink(missing_ok=True)
    return summary
