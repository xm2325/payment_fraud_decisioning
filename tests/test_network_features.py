import pandas as pd
from fraud_decisioning.features import build_features


def test_network_features_are_backward_looking():
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for i, (mins, s, r, d) in enumerate([
        (0,"C1","C9","D1"), (10,"C2","C9","D1"), (20,"C1","C8","D1")
    ]):
        rows.append(dict(transaction_id=i,timestamp=t0+pd.Timedelta(minutes=mins),day=0,type="TRANSFER",amount=10.,sender=s,recipient=r,old_balance_sender=100.,new_balance_sender=90.,account_age_days=100,device_id=d,device_change=0,country_mismatch=0,recipient_new=0,is_fraud=0,fraud_type="legitimate"))
    out,_ = build_features(pd.DataFrame(rows))
    assert out.loc[0,"device_unique_senders_24h"] == 0
    assert out.loc[1,"device_unique_senders_24h"] == 1
    assert out.loc[2,"device_unique_senders_24h"] == 2
    assert out.loc[2,"sender_unique_recipients_24h"] == 1
    assert out.loc[2,"pair_tx_24h"] == 0
