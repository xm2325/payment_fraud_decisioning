import pandas as pd
from fraud_decisioning.features import build_features

def test_velocity_is_past_only():
    df = pd.DataFrame({
        "transaction_id": [1,2,3],
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z","2026-01-01T00:30:00Z","2026-01-01T02:00:00Z"]),
        "type": ["TRANSFER"]*3,
        "amount": [10.,20.,30.],
        "sender": ["A"]*3,
        "recipient": ["B"]*3,
        "old_balance_sender": [100.]*3,
        "new_balance_sender": [90.,80.,70.],
        "account_age_days": [100]*3,
        "device_id": ["D"]*3,
        "device_change": [0]*3,
        "country_mismatch": [0]*3,
        "recipient_new": [0]*3,
        "is_fraud": [0]*3,
        "fraud_type": ["legitimate"]*3,
    })
    out, _ = build_features(df)
    assert out.loc[0, "sender_tx_1h"] == 0
    assert out.loc[1, "sender_tx_1h"] == 1
    assert out.loc[2, "sender_tx_1h"] == 0


def test_velocity_handles_microsecond_timestamp_resolution():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    df = pd.DataFrame({"transaction_id":[1,2,3],"timestamp":pd.Series(pd.to_datetime([t0,t0+pd.Timedelta(minutes=30),t0+pd.Timedelta(hours=2)])).dt.as_unit("us"),"type":["TRANSFER"]*3,"amount":[10.,20.,30.],"sender":["A"]*3,"recipient":["B"]*3,"old_balance_sender":[100.]*3,"new_balance_sender":[90.,80.,70.],"account_age_days":[100]*3,"device_id":["D"]*3,"device_change":[0]*3,"country_mismatch":[0]*3,"recipient_new":[0]*3,"is_fraud":[0]*3,"fraud_type":["legitimate"]*3})
    out,_=build_features(df)
    assert out.loc[1,"sender_tx_1h"]==1
    assert out.loc[2,"sender_tx_1h"]==0
