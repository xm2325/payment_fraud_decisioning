from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .simulate import simulate_payments
from .features import build_features
from .modeling import fit_logistic, fit_lightgbm, fit_sigmoid_calibrator, calibrate
from .evaluation import threshold_at_fpr, model_metrics, policy_grid
from .novelty import fit_novelty, anomaly_score, fit_tail_detector, tail_score, threshold_at_legit_fpr
from .experiment import two_proportion_sample_size
from .policy_sensitivity import policy_sensitivity

VELOCITY = {"sender_tx_1h","sender_tx_24h","sender_amount_24h","recipient_fanin_24h","device_activity_24h","amount_vs_7d_mean"}
NETWORK = {"sender_unique_recipients_24h","recipient_unique_senders_24h","device_unique_senders_24h","pair_tx_24h"}


def run_core(n: int, root: Path, tab: Path, model_dir: Path) -> dict:
    print(f"[1/16] Simulating {n:,} time-ordered transactions")
    raw = simulate_payments(n=n, seed=42, days=60)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    raw.to_csv(processed / "synthetic_transactions.csv.gz", index=False, compression="gzip")

    print("[2/16] Building point-in-time transaction + network features")
    df, features = build_features(raw)
    train = df[df.day < 36].copy(); val = df[(df.day >= 36) & (df.day < 48)].copy(); test = df[df.day >= 48].copy()
    novel_name = "novel_shared_device_microburst"
    assert not train.fraud_type.eq(novel_name).any() and not val.fraud_type.eq(novel_name).any() and test.fraud_type.eq(novel_name).any()
    pd.DataFrame([
        {"split":"train","n":len(train),"fraud_rate":train.is_fraud.mean(),"start":train.timestamp.min(),"end":train.timestamp.max()},
        {"split":"validation","n":len(val),"fraud_rate":val.is_fraud.mean(),"start":val.timestamp.min(),"end":val.timestamp.max()},
        {"split":"test","n":len(test),"fraud_rate":test.is_fraud.mean(),"start":test.timestamp.min(),"end":test.timestamp.max()},
    ]).to_csv(tab / "split_summary.csv", index=False)

    static = [c for c in features if c not in VELOCITY | NETWORK]
    velocity = [c for c in features if c not in NETWORK]
    model_features = velocity
    Xtr, ytr = train[model_features], train.is_fraud; Xv, yv = val[model_features], val.is_fraud; Xt, yt = test[model_features], test.is_fraud

    print("[3/16] Fitting logistic and LightGBM models")
    logit = fit_logistic(Xtr, ytr); lgbm = fit_lightgbm(Xtr, ytr)
    pv_raw = lgbm.predict_proba(Xv)[:,1]; cal = fit_sigmoid_calibrator(pv_raw, yv)
    p_logit = logit.predict_proba(Xt)[:,1]; p_lgbm = calibrate(cal, lgbm.predict_proba(Xt)[:,1]); pv_lgbm = calibrate(cal, pv_raw)
    t_logit = threshold_at_fpr(yv, logit.predict_proba(Xv)[:,1], .01); t_lgbm = threshold_at_fpr(yv, pv_lgbm, .01)
    metrics = pd.DataFrame([model_metrics("logistic",yt,p_logit,test.amount,t_logit), model_metrics("lightgbm_calibrated",yt,p_lgbm,test.amount,t_lgbm)])
    metrics.to_csv(tab / "model_metrics.csv", index=False)
    known_mask = ~test.fraud_type.eq(novel_name)
    rows=[]
    for label, cols in [("static_only",static),("static_plus_velocity",velocity),("static_plus_velocity_network",features)]:
        m=fit_lightgbm(train[cols],ytr); pv=m.predict_proba(val[cols])[:,1]; c=fit_sigmoid_calibrator(pv,yv)
        pt=calibrate(c,m.predict_proba(test[cols])[:,1]); tv0=threshold_at_fpr(yv,calibrate(c,pv),.01)
        rows += [model_metrics(label+"__all_test",yt,pt,test.amount,tv0), model_metrics(label+"__known_only",yt[known_mask],pt[known_mask],test.loc[known_mask,"amount"],tv0)]
    pd.DataFrame(rows).to_csv(tab / "feature_ablation.csv", index=False); pd.DataFrame(rows).to_csv(tab / "behavioural_feature_ablation.csv", index=False)

    print("[4/16] Optimising fraud-loss / customer-friction policy on validation")
    grid_val=policy_grid(yv,pv_lgbm,val.amount); best=grid_val.loc[grid_val.total_policy_cost.idxmin()].to_dict(); grid_test=policy_grid(yt,p_lgbm,test.amount)
    grid_test.to_csv(tab / "policy_frontier_test.csv",index=False); pd.DataFrame([best]).to_csv(tab / "best_policy_validation.csv",index=False)
    rt,bt=float(best["review_threshold"]),float(best["block_threshold"]); block=p_lgbm>=bt; review=(p_lgbm>=rt)&~block
    legit=yt.to_numpy()==0; fraud=yt.to_numpy()==1; amounts=test.amount.to_numpy()
    prevented=amounts[fraud&block].sum()*.95 + amounts[fraud&review].sum()*.65
    policy={"review_threshold":rt,"block_threshold":bt,"test_fraud_value_prevented_rate":prevented/amounts[fraud].sum(),"test_legitimate_friction_rate":((block|review)&legit).sum()/legit.sum(),"test_review_rate":review.mean(),"test_block_rate":block.mean()}
    pd.DataFrame([policy]).to_csv(tab / "policy_summary_test.csv",index=False)
    scenarios=policy_sensitivity(yv,pv_lgbm,val.amount,yt,p_lgbm,test.amount); scenarios.to_csv(tab / "policy_assumption_sensitivity.csv",index=False)
    eligible=(pv_lgbm>=rt)&(pv_lgbm<bt); pc=float(yv.to_numpy()[eligible].mean()) if eligible.any() else float(yv.mean()); pt=max(1e-6,pc*.75)
    pd.DataFrame([{"eligible_definition":"validation transactions with calibrated risk in review band","eligible_n":int(eligible.sum()),"baseline_fraud_rate":pc,"target_relative_reduction":.25,"target_fraud_rate":pt,"alpha_two_sided":.05,"power":.80,"required_n_per_arm":two_proportion_sample_size(pc,pt,.05,.80),"primary_metric":"fraud occurrence among review-band transactions","guardrail_note":"legitimate-customer completion/abandonment needs real product event data and is not simulated here"}]).to_csv(tab / "intervention_experiment_design.csv",index=False)

    print("[5/16] Sizing fraud typologies")
    typ=test[test.is_fraud==1].groupby("fraud_type").agg(transactions=("transaction_id","size"),fraud_value=("amount","sum"),mean_amount=("amount","mean")).reset_index()
    typ["transaction_share"]=typ.transactions/typ.transactions.sum(); typ["value_share"]=typ.fraud_value/typ.fraud_value.sum(); typ=typ.sort_values("fraud_value",ascending=False); typ.to_csv(tab / "fraud_typology_sizing.csv",index=False)

    print("[6/16] Training novelty detector on legitimate historical transactions")
    train_legit=train[train.is_fraud==0]; scaler,iso=fit_novelty(train_legit); sv=anomaly_score(scaler,iso,val); st=anomaly_score(scaler,iso,test); nt=threshold_at_legit_fpr(sv,yv,.01)
    tail_scales=fit_tail_detector(train_legit); tv=tail_score(tail_scales,val); tt=tail_score(tail_scales,test); tail_t=threshold_at_legit_fpr(tv,yv,.01)
    sup=p_lgbm>=t_lgbm; ano=st>=nt; tail=tt>tail_t; novel=test.fraud_type.eq(novel_name).to_numpy(); known=(test.is_fraud.eq(1)&~test.fraud_type.eq(novel_name)).to_numpy()
    novelty=pd.DataFrame([
        {"detector":"supervised_1pct_fpr","known_fraud_recall":sup[known].mean(),"novel_fraud_recall":sup[novel].mean(),"legit_flag_rate":sup[legit].mean()},
        {"detector":"isolation_forest_1pct_fpr","known_fraud_recall":ano[known].mean(),"novel_fraud_recall":ano[novel].mean(),"legit_flag_rate":ano[legit].mean()},
        {"detector":"tail_velocity_1pct_fpr","known_fraud_recall":tail[known].mean(),"novel_fraud_recall":tail[novel].mean(),"legit_flag_rate":tail[legit].mean()},
        {"detector":"hybrid_supervised_tail","known_fraud_recall":(sup|tail)[known].mean(),"novel_fraud_recall":(sup|tail)[novel].mean(),"legit_flag_rate":(sup|tail)[legit].mean()},
    ]); novelty.to_csv(tab / "novelty_detection_metrics.csv",index=False)
    joblib.dump({"model":lgbm,"calibrator":cal,"features":model_features,"review_threshold":rt,"block_threshold":bt},model_dir/"fraud_model.joblib")
    return locals()
