from __future__ import annotations
import numpy as np
import pandas as pd
from .review_queue import queue_metrics, investigation_queue, exploration_sensitivity
from .backtesting import rolling_backtest, delayed_label_retraining
from .feedback import analyst_feedback_curve
from .verification_bias import verification_bias_sensitivity
from .base_rate import prevalence_sensitivity
from .prior_shift import prior_shift_sensitivity
from .queue_sla import queue_sla_stress
from .adaptive_routing import adaptive_capacity_routing
from .monitoring import weekly_monitor


def run_operational(ctx: dict, tab) -> dict:
    df, val, test = ctx['df'], ctx['val'], ctx['test']
    model_features = ctx['model_features']; p_lgbm = ctx['p_lgbm']; pv_lgbm = ctx['pv_lgbm']
    t_lgbm = ctx['t_lgbm']; tt = ctx['tt']; tv = ctx['tv']; tail_t = ctx['tail_t']
    rt, bt = ctx['rt'], ctx['bt']; yt = ctx['yt']

    print('[7/16] Evaluating fixed-capacity fraud investigation queues')
    review_capacity = queue_metrics(test, p_lgbm, tt, capacities=(25, 50, 100, 200))
    review_capacity.to_csv(tab/'review_capacity_metrics.csv', index=False)
    investigation_queue(test, p_lgbm, tt, n=100).to_csv(tab/'sample_investigation_queue.csv', index=False)
    queue_sensitivity = exploration_sensitivity(test, p_lgbm, tt, capacities=(100, 200))
    queue_sensitivity.to_csv(tab/'exploration_share_sensitivity.csv', index=False)

    print('[8/16] Running rolling temporal backtest and delayed-label diagnostic')
    backtest = rolling_backtest(df, model_features, folds=((30,36,42),(36,42,48),(42,48,54),(48,54,60)))
    backtest.to_csv(tab/'rolling_temporal_backtest.csv', index=False)
    delayed = delayed_label_retraining(df, model_features, score_start_day=54, label_delay_days=7)
    delayed.to_csv(tab/'delayed_label_retraining.csv', index=False)

    print('[9/16] Measuring analyst-feedback recovery after anomaly discovery')
    feedback = analyst_feedback_curve(df, model_features)
    feedback.to_csv(tab/'analyst_feedback_curve.csv', index=False)

    print('[10/16] Stress-testing verification bias and random audit coverage')
    verification_raw, verification = verification_bias_sensitivity(df, model_features)
    verification_raw.to_csv(tab/'verification_bias_raw.csv', index=False)
    verification.to_csv(tab/'verification_bias_sensitivity.csv', index=False)

    print('[11/16] Translating detector rates across fraud base-rate scenarios')
    sup_flag, tail_flag = ctx['sup'], ctx['tail']
    prev_sup = prevalence_sensitivity(yt, sup_flag); prev_sup['detector']='supervised'
    prev_tail = prevalence_sensitivity(yt, tail_flag); prev_tail['detector']='tail_anomaly'
    prevalence = pd.concat([prev_sup, prev_tail], ignore_index=True)
    prevalence.to_csv(tab/'fraud_prevalence_precision_sensitivity.csv', index=False)

    print('[12/16] Stress-testing probability calibration under fraud prior shift')
    prior_raw, prior_summary = prior_shift_sensitivity(yt, p_lgbm, source_prevalence=float(val.is_fraud.mean()), target_prevalences=(.001,.0025,.005,.01,.02), repeats=20, seed=42)
    prior_raw.to_csv(tab/'prior_shift_calibration_raw.csv', index=False)
    prior_summary.to_csv(tab/'prior_shift_calibration.csv', index=False)

    print('[13/16] Stress-testing Fraud Ops queue saturation')
    queue_sla = queue_sla_stress(test, p_lgbm, tt, review_threshold=rt, block_threshold=bt, anomaly_threshold=tail_t, analyst_capacities_per_hour=(4.,6.,8.), volume_multipliers=(.5,1.,1.5,2.,4.), exploration_share=.20)
    queue_sla.to_csv(tab/'queue_sla_stress.csv', index=False)

    print('[14/16] Applying backlog-aware capacity admission control')
    adaptive = adaptive_capacity_routing(test, p_lgbm, tt, review_threshold=rt, block_threshold=bt, anomaly_threshold=tail_t, analyst_capacity_per_hour=6., traffic_multipliers=(1.,1.5,2.,4.), exploration_share=.20)
    adaptive.to_csv(tab/'adaptive_capacity_routing.csv', index=False)

    print('[15/16] Building monitoring table')
    monitor_df = pd.concat([val, test], ignore_index=True)
    monitor_prob = np.concatenate([pv_lgbm, p_lgbm]); monitor_tail=np.concatenate([tv,tt])
    monitoring = weekly_monitor(monitor_df, monitor_prob, pv_lgbm, t_lgbm, tail_scores=monitor_tail, tail_threshold=tail_t, baseline_tail_alert_rate=float((tv>tail_t).mean()))
    monitoring.to_csv(tab/'weekly_monitoring.csv', index=False)
    return locals()
