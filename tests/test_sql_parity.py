from pathlib import Path
import sqlite3
import pandas as pd
from fraud_decisioning.features import build_features


def _fixture():
    ts = pd.to_datetime([
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:30:00Z",
        "2026-01-01T00:30:00Z",  # same timestamp: must not see row 2
        "2026-01-01T02:00:00Z",
    ])
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4], "timestamp": ts, "day": [0]*4,
        "type": ["TRANSFER"]*4, "amount": [10.,20.,30.,40.],
        "sender": ["A","A","A","A"], "recipient": ["R1","R1","R1","R2"],
        "old_balance_sender": [100.]*4, "new_balance_sender": [90.,80.,70.,60.],
        "account_age_days": [100]*4, "device_id": ["D1"]*4,
        "device_change": [0]*4, "country_mismatch": [0]*4, "recipient_new": [0]*4,
        "is_fraud": [0]*4, "fraud_type": ["legitimate"]*4,
    })


def test_sql_and_python_point_in_time_parity():
    df = _fixture()
    py, _ = build_features(df)
    sql_df = pd.DataFrame({
        "transaction_id": df.transaction_id,
        "event_ts": (df.timestamp.dt.as_unit("ns").astype("int64") // 10**9).astype(int),
        "sender": df.sender, "recipient": df.recipient, "device_id": df.device_id,
        "amount": df.amount,
    })
    conn = sqlite3.connect(":memory:")
    sql_df.to_sql("transactions", conn, index=False)
    sql = (Path(__file__).resolve().parents[1] / "sql" / "sqlite_point_in_time_features.sql").read_text()
    got = pd.read_sql_query(sql, conn)
    cols = ["sender_tx_1h","sender_tx_24h","recipient_fanin_24h","device_activity_24h"]
    for c in cols:
        assert got[c].tolist() == py[c].astype(int).tolist()
    assert py.loc[1, "sender_tx_1h"] == 1
    assert py.loc[2, "sender_tx_1h"] == 1  # same-time row 2 is invisible to row 3
