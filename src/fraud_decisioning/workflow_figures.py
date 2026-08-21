from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def _save(fig_dir, name):
    plt.tight_layout(); plt.savefig(fig_dir/name, dpi=180, bbox_inches='tight'); plt.close()


def build_figures(core: dict, ops: dict, fig_dir) -> None:
    test=core['test']; yt=core['yt']; p=core['p_lgbm']; grid=core['grid_test']; policy=core['policy']
    typ=core['typ']; novelty=core['novelty']; monitoring=ops['monitoring']; review=ops['review_capacity']
    backtest=ops['backtest']; delayed=ops['delayed']; qs=ops['queue_sensitivity']; feedback=ops['feedback']
    verification=ops['verification']; prevalence=ops['prevalence']; scenarios=core['scenarios']; prior=ops['prior_summary']
    queue=ops['queue_sla']; adaptive=ops['adaptive']

    precision, recall, _ = precision_recall_curve(yt,p)
    plt.figure(figsize=(6.4,4.4)); plt.plot(recall,precision); plt.xlabel('Fraud recall'); plt.ylabel('Precision'); plt.title('Precision-recall on future test period'); _save(fig_dir,'01_precision_recall.png')
    f=grid.sort_values('legit_friction_rate'); plt.figure(figsize=(6.4,4.4)); plt.scatter(f.legit_friction_rate*100,f.fraud_value_prevented_rate*100,s=12,alpha=.45); plt.scatter([policy['test_legitimate_friction_rate']*100],[policy['test_fraud_value_prevented_rate']*100],s=70,marker='x'); plt.xlabel('Legitimate transactions challenged / blocked (%)'); plt.ylabel('Fraud value prevented (%)'); plt.title('Loss reduction vs customer friction'); _save(fig_dir,'02_loss_vs_friction.png')
    plt.figure(figsize=(7.2,4.6)); plt.barh(typ.fraud_type,typ.value_share*100); plt.xlabel('Share of fraud value (%)'); plt.title('Fraud typology sizing in test period'); plt.gca().invert_yaxis(); _save(fig_dir,'03_typology_value_share.png')
    nm=novelty.set_index('detector'); x=np.arange(len(nm)); w=.34; plt.figure(figsize=(7.6,4.4)); plt.bar(x-w/2,nm.known_fraud_recall*100,width=w,label='Known fraud'); plt.bar(x+w/2,nm.novel_fraud_recall*100,width=w,label='Unseen fraud'); plt.xticks(x,['Supervised','IsoForest','Tail','Hybrid']); plt.ylabel('Recall (%)'); plt.title('Known vs unseen attack detection'); plt.legend(); _save(fig_dir,'04_novel_attack_detection.png')
    plt.figure(figsize=(7.6,4.6)); plt.plot(monitoring.week,monitoring.model_alert_rate*100,marker='o',label='Supervised alert rate'); plt.plot(monitoring.week,monitoring.tail_alert_rate*100,marker='o',label='Tail-anomaly alert rate'); plt.plot(monitoring.week,monitoring.novel_fraud_rate*100,marker='o',label='Novel fraud rate'); plt.ylabel('Transactions (%)'); plt.xlabel('Week'); plt.title('Early warning: model alerts vs anomaly alerts'); plt.xticks(rotation=25,ha='right'); plt.legend(); _save(fig_dir,'05_monitoring_emerging_fraud.png')
    plt.figure(figsize=(7.2,4.5));
    for q,g in review.groupby('queue'): plt.plot(g.capacity_per_10k,g.fraud_value_recall*100,marker='o',label=f'{q}: fraud value')
    plt.xlabel('Analyst review capacity per 10,000 transactions'); plt.ylabel('Fraud value recalled (%)'); plt.title('Investigation value under a fixed review budget'); plt.legend(); _save(fig_dir,'06_review_capacity.png')
    plt.figure(figsize=(7.2,4.5)); plt.plot(backtest.test_days,backtest.pr_auc,marker='o',label='PR-AUC'); plt.plot(backtest.test_days,backtest.fraud_value_recall,marker='o',label='Fraud-value recall @ ~1% legit flags'); plt.xticks(rotation=25,ha='right'); plt.ylim(0,1.02); plt.ylabel('Metric'); plt.title('Rolling-origin temporal backtest'); plt.legend(); _save(fig_dir,'07_temporal_backtest.png')
    plt.figure(figsize=(6.8,4.3)); plt.bar(delayed.training_view.tolist(),delayed.novel_fraud_recall*100); plt.ylabel('Novel fraud recall (%)'); plt.title('Why fraud-label latency changes retraining conclusions'); plt.xticks(rotation=15,ha='right'); _save(fig_dir,'08_label_delay.png')
    plt.figure(figsize=(7,4.4));
    for cap,g in qs.groupby('capacity_per_10k'): plt.plot(g.exploration_share*100,g.fraud_value_recall*100,marker='o',label=f'Value recall @ {cap}/10k'); plt.plot(g.exploration_share*100,g.novel_fraud_recall*100,marker='x',linestyle='--',label=f'Novel recall @ {cap}/10k')
    plt.xlabel('Review capacity reserved for anomaly exploration (%)'); plt.ylabel('Recall (%)'); plt.title('Exploration quota is a governance trade-off'); plt.legend(fontsize=8); _save(fig_dir,'09_exploration_sensitivity.png')
    plt.figure(figsize=(7,4.4)); plt.plot(feedback.analyst_review_budget,feedback.future_novel_fraud_recall*100,marker='o',label='Future novel-fraud recall'); plt.plot(feedback.analyst_review_budget,feedback.future_fraud_value_recall*100,marker='o',label='Future fraud-value recall'); plt.xlabel('Expedited analyst reviews'); plt.ylabel('Recall (%)'); plt.title('Analyst feedback closes the emerging-fraud loop'); plt.legend(); _save(fig_dir,'10_analyst_feedback_loop.png')
    plt.figure(figsize=(7,4.4)); plt.plot(verification.audit_rate*100,verification.known_test_pr_auc_mean*100,marker='o',label='Known-fraud PR-AUC'); plt.plot(verification.audit_rate*100,verification.recall_mule_cashout_mean*100,marker='o',label='Mule-cashout recall'); plt.plot(verification.audit_rate*100,verification.recall_transfer_burst_mean*100,marker='o',label='Transfer-burst recall'); plt.xlabel('Random audit outside risk-triggered follow-up (%)'); plt.ylabel('Metric (%)'); plt.title('Investigation-driven labels can under-cover typologies'); plt.legend(); _save(fig_dir,'11_verification_bias.png')
    plt.figure(figsize=(7,4.4));
    for det,g in prevalence.groupby('detector'): plt.plot(g.fraud_prevalence*100,g.expected_alert_precision*100,marker='o',label=det)
    plt.xlabel('Fraud prevalence (%)'); plt.ylabel('Expected alert precision (%)'); plt.title('Expected precision changes with base rate'); plt.legend(); _save(fig_dir,'12_prevalence_precision.png')
    x=np.arange(len(scenarios)); plt.figure(figsize=(7.2,4.5)); plt.scatter(x,scenarios.test_fraud_value_prevented_rate*100,marker='o',label='Fraud value prevented'); plt.scatter(x,scenarios.test_legit_friction_rate*100,marker='x',label='Legitimate friction'); plt.xticks(x,scenarios.scenario,rotation=20,ha='right'); plt.ylabel('Transactions / value (%)'); plt.title('Policy sensitivity'); plt.legend(); _save(fig_dir,'13_policy_assumption_sensitivity.png')
    plt.figure(figsize=(7.2,4.5));
    for method,g in prior.groupby('method'): plt.plot(g.target_prevalence*100,g.mean_predicted_risk_mean*100,marker='o',label=method)
    ref=sorted(prior.target_prevalence.unique()); plt.plot(np.asarray(ref)*100,np.asarray(ref)*100,linestyle='--',label='perfect mean calibration'); plt.xlabel('Target fraud prevalence (%)'); plt.ylabel('Mean predicted fraud risk (%)'); plt.title('Prior-shift calibration'); plt.legend(); _save(fig_dir,'14_prior_shift_calibration.png')
    plt.figure(figsize=(7.2,4.5));
    for cap,g in queue.groupby('analyst_capacity_per_hour'): plt.plot(g.traffic_multiplier,g.final_backlog_cases,marker='o',label=f'{cap:g} reviews/hour')
    plt.xlabel('Traffic multiplier'); plt.ylabel('End-of-window review backlog'); plt.title('Fraud Ops capacity stress'); plt.legend(); _save(fig_dir,'15_queue_sla_stress.png')
    plt.figure(figsize=(7.2,4.5)); plt.plot(adaptive.traffic_multiplier,adaptive.system_fraud_value_coverage_with_blocks*100,marker='o',label='Fraud-value coverage'); plt.plot(adaptive.traffic_multiplier,adaptive.system_novel_recall_with_blocks*100,marker='o',label='Novel-fraud recall'); plt.plot(adaptive.traffic_multiplier,adaptive.candidate_acceptance_rate*100,marker='x',linestyle='--',label='Candidate acceptance'); plt.xlabel('Traffic multiplier'); plt.ylabel('Coverage / accepted candidates (%)'); plt.title('Backlog-aware admission at fixed 6 reviews/hour'); plt.legend(); _save(fig_dir,'16_adaptive_capacity_routing.png')
