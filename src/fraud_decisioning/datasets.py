from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
]


CANONICAL_PAYSIM = {
    "n": 6_362_620,
    "fraud_n": 8_213,
    "step_min": 1,
    "step_max": 743,
}


def _read_paysim(path: Path, max_step: int | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """Read PaySim, optionally stopping at a contiguous time-prefix boundary.

    The standard file is ordered by `step`. Prefix mode reads in chunks and stops once
    later steps are reached, avoiding a full 494 MB materialisation for smoke tests.
    """
    if max_step is None:
        return pd.read_csv(path)
    kept = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "step" not in chunk.columns:
            raise ValueError("PaySim file is missing column: step")
        part = chunk[chunk["step"].astype(int) <= int(max_step)]
        if len(part):
            kept.append(part)
        if len(chunk) and int(chunk["step"].min()) > int(max_step):
            break
        if len(chunk) and int(chunk["step"].max()) > int(max_step):
            break
    if not kept:
        raise ValueError(f"No PaySim rows found at or before step {max_step}")
    return pd.concat(kept, ignore_index=True)


def load_paysim(path: str | Path, max_step: int | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """Load a full or time-prefix PaySim CSV and normalize it to this repo's schema.

    PaySim has no device/country/account-age fields. Neutral defaults are used so the
    same downstream interface can run, while PaySim-specific experiments should focus
    on transaction, balance, velocity and graph features.
    """
    path = Path(path)
    raw = _read_paysim(path, max_step=max_step, chunksize=chunksize)
    missing = [c for c in PAYSIM_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"PaySim file is missing columns: {missing}")
    start = pd.Timestamp("2020-01-01", tz="UTC")
    x = pd.DataFrame({
        "transaction_id": np.arange(len(raw), dtype=np.int64),
        "source_step": raw["step"].astype(int),
        "timestamp": start + pd.to_timedelta(raw["step"].astype(int), unit="h"),
        "day": ((raw["step"].astype(int) - 1) // 24).astype(int),
        "type": raw["type"].astype(str),
        "amount": raw["amount"].astype(float),
        "sender": raw["nameOrig"].astype(str),
        "recipient": raw["nameDest"].astype(str),
        "old_balance_sender": raw["oldbalanceOrg"].astype(float),
        "new_balance_sender": raw["newbalanceOrig"].astype(float),
        "account_age_days": 0,
        "device_id": raw["nameOrig"].astype(str),  # neutral proxy: no cross-account device sharing claim
        "device_change": 0,
        "country_mismatch": 0,
        "recipient_new": 0,
        "is_fraud": raw["isFraud"].astype(int),
        "fraud_type": np.where(raw["isFraud"].astype(int).eq(1), "paysim_injected_fraud", "legitimate"),
        "is_flagged_fraud": raw["isFlaggedFraud"].astype(int),
    })
    return x.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def paysim_data_audit(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "fraud_n": int(df.is_fraud.sum()),
        "fraud_rate": float(df.is_fraud.mean()),
        "start": str(df.timestamp.min()),
        "end": str(df.timestamp.max()),
        "days": int(df.day.max() - df.day.min() + 1),
    }


def canonical_paysim_status(df: pd.DataFrame) -> dict:
    """Compare a normalized full dataset with published standard PaySim counts."""
    actual = {
        "n": int(len(df)),
        "fraud_n": int(df.is_fraud.sum()),
        "step_min": int(df.source_step.min()),
        "step_max": int(df.source_step.max()),
    }
    matches = {k: actual[k] == v for k, v in CANONICAL_PAYSIM.items()}
    return {
        "actual": actual,
        "expected": dict(CANONICAL_PAYSIM),
        "matches": matches,
        "is_canonical_full": bool(all(matches.values())),
    }
