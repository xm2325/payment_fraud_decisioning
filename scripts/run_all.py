from __future__ import annotations
import json, os, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_decisioning.workflow_core import run_core
from fraud_decisioning.workflow_ops import run_operational
from fraud_decisioning.workflow_figures import build_figures

OUT=ROOT/'outputs'; TAB=OUT/'tables'; FIG=OUT/'figures'; MODEL=ROOT/'models'
for p in [TAB,FIG,MODEL,ROOT/'data'/'processed']: p.mkdir(parents=True,exist_ok=True)


def main(n=120_000):
    core=run_core(n,ROOT,TAB,MODEL)
    ops=run_operational(core,TAB)
    build_figures(core,ops,FIG)
    pd.DataFrame({'feature':core['model_features'],'importance':core['lgbm'].feature_importances_}).sort_values('importance',ascending=False).to_csv(TAB/'feature_importance.csv',index=False)
    result={
        'n_transactions':int(len(core['df'])),
        'train_n':int(len(core['train'])),'validation_n':int(len(core['val'])),'test_n':int(len(core['test'])),
        'test_fraud_rate':float(core['test'].is_fraud.mean()),'test_novel_fraud_count':int(core['novel'].sum()),
        'model_metrics':core['metrics'].to_dict(orient='records'),'policy':core['policy'],'novelty':core['novelty'].to_dict(orient='records'),
        'review_capacity':ops['review_capacity'].to_dict(orient='records'),'exploration_sensitivity':ops['queue_sensitivity'].to_dict(orient='records'),
        'rolling_backtest':ops['backtest'].to_dict(orient='records'),'delayed_label_retraining':ops['delayed'].to_dict(orient='records'),
        'analyst_feedback_curve':ops['feedback'].to_dict(orient='records'),'verification_bias_sensitivity':ops['verification'].to_dict(orient='records'),
        'fraud_prevalence_precision_sensitivity':ops['prevalence'].to_dict(orient='records'),'policy_assumption_sensitivity':core['scenarios'].to_dict(orient='records'),
        'prior_shift_calibration':ops['prior_summary'].to_dict(orient='records'),'queue_sla_stress':ops['queue_sla'].to_dict(orient='records'),
        'adaptive_capacity_routing':ops['adaptive'].to_dict(orient='records')}
    with open(OUT/'summary.json','w') as f: json.dump(result,f,indent=2)
    print('[16/16] Done'); print(json.dumps(result,indent=2))

if __name__=='__main__': main(int(os.environ.get('FRAUD_N','120000')))
