from __future__ import annotations
import numpy as np
import pandas as pd

TX_TYPES = np.array(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
TX_P = np.array([0.46, 0.21, 0.16, 0.11, 0.06])


def simulate_payments(n: int = 180_000, seed: int = 42, days: int = 60) -> pd.DataFrame:
    """Generate a transparent synthetic payment stream with known and unseen fraud patterns.

    The generator is for pipeline validation only; results must not be presented as real Moniepoint data.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    secs = np.sort(rng.integers(0, days * 86400, size=n))
    ts = start + pd.to_timedelta(secs, unit="s")
    day = (secs // 86400).astype(int)

    n_senders, n_customers, n_merchants = 14_000, 9_000, 2_500
    sender_activity = rng.lognormal(0.0, 1.0, n_senders)
    sender_activity /= sender_activity.sum()
    senders = rng.choice(n_senders, n, p=sender_activity)
    tx_type = rng.choice(TX_TYPES, n, p=TX_P)

    is_merchant = tx_type == "PAYMENT"
    recipients_num = np.where(
        is_merchant,
        rng.integers(0, n_merchants, n),
        rng.integers(0, n_customers, n),
    )
    recipient = np.where(is_merchant, "M" + recipients_num.astype(str), "C" + recipients_num.astype(str))

    amount = np.exp(rng.normal(4.0, 1.05, n)).clip(1, 25_000)
    base_balance = np.exp(rng.normal(7.4, 1.0, n)).clip(50, 250_000)
    account_age_days = rng.integers(5, 2500, n)
    device_id = rng.integers(0, 22_000, n).astype(str)
    device_change = rng.random(n) < 0.035
    country_mismatch = rng.random(n) < 0.012
    recipient_new = rng.random(n) < 0.18

    # Known fraud exists across the timeline.
    known = rng.random(n) < 0.012
    fraud_type = np.full(n, "legitimate", dtype=object)
    known_type = rng.choice(["account_takeover", "transfer_burst", "mule_cashout"], known.sum(), p=[0.46, 0.29, 0.25])
    fraud_type[known] = known_type

    compromised = rng.integers(0, 220, known.sum())
    senders[known] = compromised
    known_idx = np.where(known)[0]

    ato = fraud_type == "account_takeover"
    tx_type[ato] = "TRANSFER"
    amount[ato] *= rng.uniform(3.0, 8.0, ato.sum())
    device_change[ato] = rng.random(ato.sum()) < 0.88
    country_mismatch[ato] = rng.random(ato.sum()) < 0.48
    recipient_new[ato] = rng.random(ato.sum()) < 0.90
    device_id[ato] = np.array([f"NEW_{i}" for i in np.where(ato)[0]], dtype=object)

    burst = fraud_type == "transfer_burst"
    tx_type[burst] = "TRANSFER"
    amount[burst] *= rng.uniform(1.4, 4.0, burst.sum())
    recipient_new[burst] = rng.random(burst.sum()) < 0.78
    device_id[burst] = np.array([f"B_{i}" for i in np.where(burst)[0]], dtype=object)

    mule = fraud_type == "mule_cashout"
    tx_type[mule] = "CASH_OUT"
    amount[mule] *= rng.uniform(2.0, 6.0, mule.sum())
    recipients_num[mule] = rng.integers(0, 70, mule.sum())
    recipient[mule] = np.array([f"C{x}" for x in recipients_num[mule]], dtype=object)
    recipient_new[mule] = False
    device_id[mule] = np.array([f"MC_{i}" for i in np.where(mule)[0]], dtype=object)

    # Novel attack appears only late in the stream. It looks normal on common fraud cues,
    # but many accounts share a tiny set of devices and transact in short bursts.
    novel = (day >= 48) & (rng.random(n) < 0.010)
    known[novel] = False
    fraud_type[novel] = "novel_shared_device_microburst"
    tx_type[novel] = "PAYMENT"
    amount[novel] = np.exp(rng.normal(3.15, 0.35, novel.sum())).clip(5, 120)
    senders[novel] = rng.integers(300, 360, novel.sum())
    recipient[novel] = np.array([f"M{x}" for x in rng.integers(200, 240, novel.sum())], dtype=object)
    device_id[novel] = np.array([f"SHARED_{x}" for x in rng.integers(0, 4, novel.sum())], dtype=object)
    device_change[novel] = False
    country_mismatch[novel] = False
    recipient_new[novel] = False

    is_fraud = fraud_type != "legitimate"
    old_balance = base_balance
    new_balance = np.maximum(0.0, old_balance - amount)

    df = pd.DataFrame({
        "transaction_id": np.arange(n, dtype=np.int64),
        "timestamp": ts,
        "day": day,
        "type": tx_type,
        "amount": amount.astype(float),
        "sender": np.array([f"C{x}" for x in senders], dtype=object),
        "recipient": recipient,
        "old_balance_sender": old_balance.astype(float),
        "new_balance_sender": new_balance.astype(float),
        "account_age_days": account_age_days,
        "device_id": device_id,
        "device_change": device_change.astype(int),
        "country_mismatch": country_mismatch.astype(int),
        "recipient_new": recipient_new.astype(int),
        "is_fraud": is_fraud.astype(int),
        "fraud_type": fraud_type,
    })
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
