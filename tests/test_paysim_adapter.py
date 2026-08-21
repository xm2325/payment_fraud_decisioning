import pandas as pd
from fraud_decisioning.datasets import load_paysim


def test_paysim_adapter(tmp_path):
    p = tmp_path / "p.csv"
    pd.DataFrame([{ "step":1,"type":"TRANSFER","amount":10,"nameOrig":"C1","oldbalanceOrg":20,"newbalanceOrig":10,"nameDest":"C2","oldbalanceDest":0,"newbalanceDest":10,"isFraud":1,"isFlaggedFraud":0 }]).to_csv(p,index=False)
    df = load_paysim(p)
    assert df.loc[0,"is_fraud"] == 1
    assert df.loc[0,"sender"] == "C1"
    assert df.loc[0,"day"] == 0


def test_paysim_prefix_loader_stops_at_step(tmp_path):
    import pandas as pd
    from fraud_decisioning.datasets import load_paysim, canonical_paysim_status
    rows=[]
    for step in [1,1,2,2,3,3]:
        rows.append({"step":step,"type":"TRANSFER","amount":1.0,"nameOrig":f"C{step}a","oldbalanceOrg":1.0,"newbalanceOrig":0.0,"nameDest":f"C{step}b","oldbalanceDest":0.0,"newbalanceDest":1.0,"isFraud":int(step==2),"isFlaggedFraud":0})
    path=tmp_path/'p.csv'; pd.DataFrame(rows).to_csv(path,index=False)
    x=load_paysim(path,max_step=2,chunksize=2)
    assert x.source_step.max() == 2
    assert len(x) == 4
    assert canonical_paysim_status(x)["is_canonical_full"] is False
