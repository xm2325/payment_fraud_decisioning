from __future__ import annotations
from pathlib import Path
import base64, html
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "outputs" / "moniepoint_fraud_case_study.html"


def img(name):
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(name)}">'


def fmt_table(df, digits=3, pct_cols=()):
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = (d[c] * 100).map(lambda x: f"{x:.1f}%")
    for c in d.select_dtypes(include="number").columns:
        if c not in pct_cols:
            d[c] = d[c].map(lambda x: f"{x:.{digits}f}")
    return d.to_html(index=False, border=0, classes="data")


ab = pd.read_csv(TAB / "feature_ablation.csv")
policy = pd.read_csv(TAB / "policy_summary_test.csv").iloc[0]
nov = pd.read_csv(TAB / "novelty_detection_metrics.csv")
typ = pd.read_csv(TAB / "fraud_typology_sizing.csv")
exp = pd.read_csv(TAB / "intervention_experiment_design.csv").iloc[0]
mon = pd.read_csv(TAB / "weekly_monitoring.csv")
queue = pd.read_csv(TAB / "review_capacity_metrics.csv")
back = pd.read_csv(TAB / "rolling_temporal_backtest.csv")
delay = pd.read_csv(TAB / "delayed_label_retraining.csv")
feedback = pd.read_csv(TAB / "analyst_feedback_curve.csv")
verify = pd.read_csv(TAB / "verification_bias_sensitivity.csv")
prevalence = pd.read_csv(TAB / "fraud_prevalence_precision_sensitivity.csv")
policy_sens = pd.read_csv(TAB / "policy_assumption_sensitivity.csv")
prior_shift = pd.read_csv(TAB / "prior_shift_calibration.csv")
queue_sla = pd.read_csv(TAB / "queue_sla_stress.csv")
adaptive = pd.read_csv(TAB / "adaptive_capacity_routing.csv")

PAYSIM = ROOT / "results" / "paysim_full"
paysim_section = ""
if (PAYSIM / "model_ablation.csv").exists() and (PAYSIM / "split_summary.csv").exists():
    pab = pd.read_csv(PAYSIM / "model_ablation.csv")
    psp = pd.read_csv(PAYSIM / "split_summary.csv")
    ptx = pab[pab.model.eq("transaction_only")].iloc[0]
    ph = pab[pab.model.eq("transaction_plus_history")].iloc[0]
    prel = pab[pab.model.eq("transaction_plus_relational")].iloc[0] if pab.model.eq("transaction_plus_relational").any() else ph
    pbal = pab[pab.model.eq("full_with_simulator_balances")].iloc[0]
    paysim_section = f"""<section><h2>External validation: full 6.36M-row PaySim</h2><p>A GitHub-hosted run verified all 6,362,620 transactions and 8,213 fraud labels, then used DuckDB strict prior-step features and a future split. Transaction-only PR-AUC is <strong>{ptx.pr_auc:.3f}</strong>; basic history is <strong>{ph.pr_auc:.3f}</strong>; relational pair/counterparty history reaches <strong>{prel.pr_auc:.3f}</strong>. At its validation-derived operating point the relational model has <strong>{prel.precision*100:.1f}% precision</strong>, <strong>{prel.recall*100:.1f}% recall</strong> and <strong>{prel.fraud_value_recall*100:.1f}% fraud-value recall</strong>. Adding PaySim old/new-balance derivatives raises PR-AUC to <strong>{pbal.pr_auc:.3f}</strong>, recorded as simulator-specific sensitivity rather than the headline result.</p>{fmt_table(pab[['model','pr_auc','precision','recall','legit_flag_rate','fraud_value_recall']],3,('precision','recall','legit_flag_rate','fraud_value_recall'))}<p>The temporal split changes fraud prevalence from <strong>{psp.iloc[0].fraud_rate*100:.3f}%</strong> in train to <strong>{psp.iloc[1].fraud_rate*100:.3f}%</strong> in validation and <strong>{psp.iloc[2].fraud_rate*100:.3f}%</strong> in test.</p><p class="small">PaySim is synthetic mobile-money data. Raw PaySim rows are not included in this report or repository.</p></section>"""

static = ab[ab.model.eq("static_only__known_only")].iloc[0]
velocity = ab[ab.model.eq("static_plus_velocity__known_only")].iloc[0]
network = ab[ab.model.eq("static_plus_velocity_network__known_only")].iloc[0]
tail = nov[nov.detector.eq("tail_velocity_1pct_fpr")].iloc[0]
ato = typ[typ.fraud_type.eq("account_takeover")].iloc[0]
q100_model = queue[(queue.queue.eq("model")) & (queue.capacity_per_10k.eq(100))].iloc[0]
q100_two = queue[(queue.queue.eq("two_lane_80_20")) & (queue.capacity_per_10k.eq(100))].iloc[0]
q200_model = queue[(queue.queue.eq("model")) & (queue.capacity_per_10k.eq(200))].iloc[0]
q200_two = queue[(queue.queue.eq("two_lane_80_20")) & (queue.capacity_per_10k.eq(200))].iloc[0]
delayed = delay[delay.training_view.str.startswith("asof")].iloc[0]
oracle = delay[delay.training_view.eq("oracle_instant_labels")].iloc[0]
fb0 = feedback[feedback.analyst_review_budget.eq(0)].iloc[0]
fb10 = feedback[feedback.analyst_review_budget.eq(10)].iloc[0]
fb100 = feedback[feedback.analyst_review_budget.eq(100)].iloc[0]
v0 = verify[verify.audit_rate.eq(0.0)].iloc[0]
v10 = verify[verify.audit_rate.eq(0.10)].iloc[0]
vfull = verify[verify.audit_rate.eq(1.0)].iloc[0]
p001_sup = prevalence[(prevalence.detector.eq("supervised")) & (prevalence.fraud_prevalence.eq(0.001))].iloc[0]
p001_tail = prevalence[(prevalence.detector.eq("tail_anomaly")) & (prevalence.fraud_prevalence.eq(0.001))].iloc[0]
prior_001_u = prior_shift[(prior_shift.target_prevalence.eq(0.001)) & (prior_shift.method.eq("unadjusted"))].iloc[0]
prior_001_a = prior_shift[(prior_shift.target_prevalence.eq(0.001)) & (prior_shift.method.eq("prior_adjusted"))].iloc[0]
prior_020_u = prior_shift[(prior_shift.target_prevalence.eq(0.02)) & (prior_shift.method.eq("unadjusted"))].iloc[0]
prior_020_a = prior_shift[(prior_shift.target_prevalence.eq(0.02)) & (prior_shift.method.eq("prior_adjusted"))].iloc[0]
q4_1 = queue_sla[(queue_sla.analyst_capacity_per_hour.eq(4.0)) & (queue_sla.traffic_multiplier.eq(1.0))].iloc[0]
q6_1 = queue_sla[(queue_sla.analyst_capacity_per_hour.eq(6.0)) & (queue_sla.traffic_multiplier.eq(1.0))].iloc[0]
q8_15 = queue_sla[(queue_sla.analyst_capacity_per_hour.eq(8.0)) & (queue_sla.traffic_multiplier.eq(1.5))].iloc[0]
a1 = adaptive[adaptive.traffic_multiplier.eq(1.0)].iloc[0]
a15 = adaptive[adaptive.traffic_multiplier.eq(1.5)].iloc[0]
a2 = adaptive[adaptive.traffic_multiplier.eq(2.0)].iloc[0]
a4 = adaptive[adaptive.traffic_multiplier.eq(4.0)].iloc[0]

css = '''
body{font-family:Arial,Helvetica,sans-serif;background:#f5f7fb;color:#172033;margin:0;line-height:1.55}
main{max-width:1120px;margin:auto;padding:32px 24px 80px}.hero{background:#111827;color:white;padding:34px;border-radius:18px}.hero h1{margin:0 0 8px;font-size:34px}.hero p{max-width:900px;color:#dbe3f3}
.badge{display:inline-block;background:#25334d;padding:5px 10px;border-radius:999px;font-size:12px;margin-right:6px}.warn{background:#fff6d9;border-left:5px solid #c48b00;padding:14px 16px;margin:18px 0;border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.card{background:white;padding:18px;border-radius:14px;box-shadow:0 2px 10px #0000000d}.k{font-size:29px;font-weight:700;margin-bottom:6px}
section{background:white;margin-top:18px;padding:24px;border-radius:14px;box-shadow:0 2px 10px #0000000d}h2{margin-top:0}h3{margin-bottom:5px}img{width:100%;height:auto;border:1px solid #e5e7eb;border-radius:10px;margin-top:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.data{border-collapse:collapse;width:100%;font-size:13px}.data th,.data td{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right}.data th:first-child,.data td:first-child{text-align:left}.callout{font-size:18px;font-weight:600}.small{font-size:13px;color:#5c667a}code{background:#eef2f7;padding:2px 5px;border-radius:4px}@media(max-width:800px){.cards,.grid{grid-template-columns:1fr}.hero h1{font-size:27px}}
'''

cards = f'''
<div class="cards">
<div class="card"><div class="k">{velocity.fraud_value_recall_at_1pct_fpr*100:.1f}%</div><div>known-fraud value recall after point-in-time velocity/history</div></div>
<div class="card"><div class="k">{tail.novel_fraud_recall*100:.1f}%</div><div>test-only attack recall from the label-free tail detector</div></div>
<div class="card"><div class="k">0% → {fb100.future_novel_fraud_recall*100:.1f}%</div><div>later novel recall after 100 anomaly-ranked analyst reviews</div></div>
<div class="card"><div class="k">{p001_sup.expected_alert_precision*100:.1f}%</div><div>expected supervised alert precision if fraud prevalence is 0.1%</div></div>
</div>'''

body = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Payment Fraud Decisioning Case Study v1.2</title><style>{css}</style></head><body><main>
<div class="hero"><span class="badge">Fraud Data Science</span><span class="badge">Python + SQL</span><span class="badge">Temporal evaluation</span><span class="badge">Fraud Ops</span><h1>Payment Fraud Decisioning & Early-Warning Workbench — v1.2</h1><p>A Moniepoint-aligned case study connecting transaction risk, loss/friction policy, unseen-pattern detection, analyst capacity, label maturity, investigation feedback and monitoring.</p></div>
<div class="warn"><strong>Data boundary.</strong> The main decisioning experiments use a transparent 120,000-transaction synthetic stream. A separate section reports a verified full 6.36M-row PaySim external benchmark. Neither dataset is Moniepoint data or a production impact claim.</div>
{cards}
<section><h2>1. One classifier is not the fraud system</h2><p class="callout">Known fraud, emerging fraud, customer friction, analyst capacity and label maturity are different objectives.</p><p>The pipeline uses train → validation → future test. A shared-device microburst attack starts only on day 48, after supervised training and calibration, so the case study can measure supervised failure and the path from anomaly discovery to new training evidence.</p></section>
<section><h2>2. Feature ablation changed the architecture</h2><p>Static features produce known-fraud PR-AUC <strong>{static.pr_auc:.3f}</strong>. Adding strictly backward-looking velocity/history raises it to <strong>{velocity.pr_auc:.3f}</strong> and fraud-value recall from <strong>{static.fraud_value_recall_at_1pct_fpr*100:.1f}%</strong> to <strong>{velocity.fraud_value_recall_at_1pct_fpr*100:.1f}%</strong>. Adding network-style history lowers PR-AUC to <strong>{network.pr_auc:.3f}</strong>, so those signals remain in anomaly/investigation rather than being forced into the classifier.</p>{fmt_table(ab[ab.model.str.endswith('__known_only')][['model','pr_auc','recall_at_1pct_fpr','fraud_value_recall_at_1pct_fpr','legit_flag_rate']],3,('recall_at_1pct_fpr','fraud_value_recall_at_1pct_fpr','legit_flag_rate'))}{img('01_precision_recall.png')}</section>
<section><h2>3. Risk score becomes an operational policy</h2><p>Review/block thresholds are selected on validation data. Under the stated simulated intervention-efficacy assumptions, the selected test policy prevents <strong>{policy.test_fraud_value_prevented_rate*100:.1f}%</strong> of fraud value while <strong>{policy.test_legitimate_friction_rate*100:.1f}%</strong> of legitimate transactions enter review or block.</p>{img('02_loss_vs_friction.png')}<p class="small">This is a scenario result, not a real prevented-loss claim.</p></section>
<section><h2>4. Policy impact is conditional on business assumptions</h2><p>Across four pre-specified efficacy/cost scenarios, the validation-selected block threshold remains <strong>0.25</strong>, while the review threshold moves from <strong>0.03 to 0.06</strong> when customer friction is more expensive or interventions are more conservative. Test fraud-value prevention ranges from <strong>{policy_sens.test_fraud_value_prevented_rate.min()*100:.1f}% to {policy_sens.test_fraud_value_prevented_rate.max()*100:.1f}%</strong> and legitimate friction from <strong>{policy_sens.test_legit_friction_rate.min()*100:.2f}% to {policy_sens.test_legit_friction_rate.max()*100:.2f}%</strong>. All four validation-selected threshold pairs have zero retrospective regret against the corresponding test-grid optimum in this simulator.</p>{img('13_policy_assumption_sensitivity.png')}{fmt_table(policy_sens[['scenario','selected_review_threshold','selected_block_threshold','test_fraud_value_prevented_rate','test_legit_friction_rate','test_policy_regret']],3,('test_fraud_value_prevented_rate','test_legit_friction_rate','test_policy_regret'))}<p class="small">The scenarios are sensitivity assumptions, not estimated Moniepoint costs.</p></section>
<section><h2>5. Typology sizing changes priority</h2><p>Account takeover is <strong>{ato.transaction_share*100:.1f}%</strong> of fraud transactions but <strong>{ato.value_share*100:.1f}%</strong> of fraud value. Case count and loss contribution therefore imply different investigation priorities.</p>{img('03_typology_value_share.png')}</section>
<section><h2>6. The test-only attack exposes supervised-model failure</h2><p>The supervised model gets <strong>0%</strong> recall on the attack absent from training. The label-free tail detector reaches <strong>{tail.novel_fraud_recall*100:.1f}%</strong> recall at <strong>{tail.legit_flag_rate*100:.2f}%</strong> legitimate flag rate.</p>{img('04_novel_attack_detection.png')}</section>
<section><h2>7. Fraud Ops needs a review-budget policy</h2><p>At 100 reviews per 10,000 transactions, model-only captures <strong>{q100_model.fraud_value_recall*100:.1f}%</strong> of fraud value and <strong>0%</strong> of the new attack. Fixed 80/20 routing captures <strong>{q100_two.fraud_value_recall*100:.1f}%</strong> of fraud value and <strong>{q100_two.novel_fraud_recall*100:.1f}%</strong> of the new attack. At 200/10k, it keeps <strong>{q200_two.fraud_value_recall*100:.1f}%</strong> value recall versus <strong>{q200_model.fraud_value_recall*100:.1f}%</strong> for model-only, while adding <strong>{q200_two.novel_fraud_recall*100:.1f}%</strong> new-attack recall.</p>{img('06_review_capacity.png')}{img('09_exploration_sensitivity.png')}</section>
<section><h2>8. Rolling backtest shows a metric blind spot</h2><p>Before the attack, future PR-AUC is 0.661 and 0.571. After it starts, PR-AUC falls to 0.330–0.374, while fraud-value recall remains above 81% because the new attack is deliberately low-value. A value KPI can therefore look acceptable while a new fraud family is being missed.</p>{img('07_temporal_backtest.png')}{fmt_table(back[['test_days','pr_auc','fraud_value_recall','known_fraud_recall','novel_fraud_recall','legit_flag_rate']],3,('fraud_value_recall','known_fraud_recall','novel_fraud_recall','legit_flag_rate'))}</section>
<section><h2>9. Fraud-label delay changes retraining conclusions</h2><p>For days 54–59, a 7-day mature-label view gets <strong>{delayed.novel_fraud_recall*100:.1f}%</strong> new-attack recall. An impossible instant-label oracle gets <strong>{oracle.novel_fraud_recall*100:.1f}%</strong>. The oracle quantifies temporal leakage; it is not a deployable result.</p>{img('08_label_delay.png')}{fmt_table(delay[['training_view','available_training_end_day','pr_auc','fraud_recall','novel_fraud_recall','legit_flag_rate']],3,('fraud_recall','novel_fraud_recall','legit_flag_rate'))}</section>
<section><h2>10. Anomaly review should feed supervised learning</h2><p>Without new labels, later novel recall is <strong>{fb0.future_novel_fraud_recall*100:.1f}%</strong>. Ten anomaly-ranked reviews yield ten confirmed novel cases in this controlled simulator and raise later recall to <strong>{fb10.future_novel_fraud_recall*100:.1f}%</strong>. At 100 reviews, 95 are confirmed novel cases and later recall reaches <strong>{fb100.future_novel_fraud_recall*100:.1f}%</strong>.</p>{img('10_analyst_feedback_loop.png')}{fmt_table(feedback[['analyst_review_budget','feedback_novel_fraud_n','future_pr_auc','future_novel_fraud_recall','future_fraud_value_recall','future_legit_flag_rate']],3,('future_novel_fraud_recall','future_fraud_value_recall','future_legit_flag_rate'))}<p class="small">The attack was deliberately designed to be visible to the tail detector, so confirmation yield is optimistic and must remain labelled as synthetic.</p></section>
<section><h2>11. Investigation policy changes training-label coverage</h2><p>Risk-triggered follow-up alone labels only <strong>{v0.labelled_train_rate_mean*100:.1f}%</strong> of the historical stream and produces known-fraud PR-AUC <strong>{v0.known_test_pr_auc_mean:.3f}</strong> with mule-cashout recall <strong>{v0.recall_mule_cashout_mean*100:.1f}%</strong>. Full historical labels give <strong>{vfull.known_test_pr_auc_mean:.3f}</strong> and <strong>{vfull.recall_mule_cashout_mean*100:.1f}%</strong>. With a 10% random audit outside the risk-triggered set, mean PR-AUC is <strong>{v10.known_test_pr_auc_mean:.3f}</strong>.</p>{img('11_verification_bias.png')}{fmt_table(verify[['audit_rate','labelled_train_rate_mean','labelled_fraud_rate_mean','known_test_pr_auc_mean','recall_transfer_burst_mean','recall_mule_cashout_mean']],3,('audit_rate','labelled_train_rate_mean','labelled_fraud_rate_mean','recall_transfer_burst_mean','recall_mule_cashout_mean'))}</section>
<section><h2>12. Precision depends on the deployment base rate</h2><p>Holding measured TPR/FPR fixed, Bayes' rule gives expected precision of only <strong>{p001_sup.expected_alert_precision*100:.1f}%</strong> for the supervised detector and <strong>{p001_tail.expected_alert_precision*100:.1f}%</strong> for the tail detector when fraud prevalence is 0.1%. This is why synthetic observed precision cannot be copied into production staffing or customer-friction estimates.</p>{img('12_prevalence_precision.png')}{fmt_table(prevalence[['detector','fraud_prevalence','expected_alert_precision']],3,('fraud_prevalence','expected_alert_precision'))}</section>
<section><h2>13. Prior shift changes the meaning of a calibrated probability</h2><p>The validation calibrator was fitted at a fraud prevalence of about 1.07%. If deployment prevalence is only 0.10%, leaving the posterior unchanged gives mean predicted risk <strong>{prior_001_u.mean_predicted_risk_mean*100:.3f}%</strong> and Brier <strong>{prior_001_u.brier_mean:.6f}</strong>. A prior-probability correction reduces these to <strong>{prior_001_a.mean_predicted_risk_mean*100:.3f}%</strong> and <strong>{prior_001_a.brier_mean:.6f}</strong>. At a 2% target prior, however, correction slightly worsens Brier (<strong>{prior_020_u.brier_mean:.6f}</strong> → <strong>{prior_020_a.brier_mean:.6f}</strong>), which is a warning that pure label shift is not enough when the future period also contains concept drift.</p>{img('14_prior_shift_calibration.png')}{fmt_table(prior_shift[['target_prevalence','method','mean_predicted_risk_mean','brier_mean','log_loss_mean','ece_10bin_mean']],4,('target_prevalence','mean_predicted_risk_mean'))}<p class="small">Prior correction assumes stable class-conditional feature distributions. It is a sensitivity calculation, not a substitute for deployment calibration monitoring.</p></section>
<section><h2>14. Fraud Ops can fail even when the model still scores correctly</h2><p>The current policy plus exploration channel generates about <strong>{q4_1.observed_candidate_arrival_rate_per_hour:.2f} review candidates/hour</strong> in the reference test stream. At four reviews/hour, 1× traffic has utilisation <strong>{q4_1.capacity_utilisation*100:.1f}%</strong> and ends the 12-day window with <strong>{q4_1.final_backlog_cases:.0f}</strong> queued cases. Six reviews/hour brings utilisation below 100% and the maximum wait proxy to <strong>{q6_1.max_wait_proxy_hours:.2f} hours</strong>, meeting the four-hour stress-test proxy at 1× traffic. Even eight reviews/hour fails the proxy at 1.5× traffic, with a <strong>{q8_15.final_backlog_cases:.1f}</strong>-case end backlog.</p>{img('15_queue_sla_stress.png')}{fmt_table(queue_sla[['traffic_multiplier','analyst_capacity_per_hour','scaled_arrival_rate_per_hour','capacity_utilisation','final_backlog_cases','max_wait_proxy_hours','meets_4h_sla_proxy']],3,('capacity_utilisation',))}<p class="small">The four-hour target and staffing levels are scenario assumptions. The point is to connect alert policy to operational capacity rather than claim Moniepoint staffing needs.</p></section>
<section><h2>15. Capacity monitoring now closes the loop</h2><p>v1.0 added backlog-aware admission control rather than only reporting overload. With a fixed six-review/hour capacity, the controller accepts <strong>{a1.candidate_acceptance_rate*100:.1f}%</strong> of candidates at 1× traffic and <strong>{a15.candidate_acceptance_rate*100:.1f}%</strong> at 1.5×. System fraud-value coverage from automatic blocks plus timely analyst admissions changes from <strong>{a1.system_fraud_value_coverage_with_blocks*100:.1f}%</strong> to <strong>{a15.system_fraud_value_coverage_with_blocks*100:.1f}%</strong>. At 4× traffic, candidate acceptance falls to <strong>{a4.candidate_acceptance_rate*100:.1f}%</strong> and novel-fraud recall to <strong>{a4.system_novel_recall_with_blocks*100:.1f}%</strong>, making the cost of protecting the SLA visible rather than hiding it in backlog.</p>{img('16_adaptive_capacity_routing.png')}{fmt_table(adaptive[['traffic_multiplier','candidate_acceptance_rate','capacity_utilisation_after_admission','system_fraud_value_coverage_with_blocks','system_novel_recall_with_blocks','legitimate_review_rate','median_dynamic_exploit_cutoff']],3,('candidate_acceptance_rate','capacity_utilisation_after_admission','system_fraud_value_coverage_with_blocks','system_novel_recall_with_blocks','legitimate_review_rate'))}<p class="small">Traffic multipliers keep the source-stream score/class mix fixed and scale effective review demand. This is an operational sensitivity, not a production traffic forecast.</p></section>
<section><h2>16. Monitoring needs signals that do not wait for labels</h2><p>When the attack begins, model-score PSI stays close to zero while tail-anomaly alert rate rises enough to move status to <strong>INVESTIGATE</strong>. Fraud Ops therefore receives a signal before mature labels can support retraining.</p>{img('05_monitoring_emerging_fraud.png')}{fmt_table(mon[['week','model_alert_rate','tail_alert_rate','novel_fraud_rate','score_psi_vs_validation','status']],4,('model_alert_rate','tail_alert_rate','novel_fraud_rate'))}</section>
<section><h2>17. Experiment design without a fabricated effect</h2><p>The validation review band has <strong>{exp.baseline_fraud_rate*100:.2f}%</strong> fraud incidence. Detecting a 25% relative reduction with two-sided alpha 0.05 and 80% power requires about <strong>{int(exp.required_n_per_arm):,} transactions per arm</strong>. Customer completion/abandonment would be a guardrail, but no effect is claimed because the current data do not contain that product outcome.</p></section>
<section><h2>18. SQL and Python share one point-in-time contract</h2><p>Equal-timestamp transactions are processed as a batch so ordering inside a timestamp cannot leak history. A standard-library SQLite parity test executes the reference SQL and checks that sender velocity, recipient fan-in and device activity match the Python feature builder.</p></section>
{paysim_section}<section><h2>19. Production-shaped hand-off</h2><div class="grid"><div><h3>Data and features</h3><p>Point-in-time Python features, executable SQL parity fixture and a PaySim adapter.</p><h3>Models</h3><p>Logistic baseline, calibrated LightGBM, Isolation Forest and an interpretable upper-tail detector.</p></div><div><h3>Operations</h3><p>Approve/review/block policy, fixed analyst budgets, exploit-explore routing, reason codes and analyst feedback.</p><h3>Engineering</h3><p>Reproducible runner, 22 tests, CI smoke run, Docker, model artifact and FastAPI policy endpoint.</p></div></div></section>
<section><h2>20. What real payment data would add</h2><p>Confirmed chargeback timestamps, investigation outcomes, device fingerprint quality, merchant/account network history, customer challenge completion, review capacity, intervention assignment and production base rates. These are required before estimating real fraud loss, operational precision or causal treatment effects.</p></section>
</main></body></html>'''
OUT.write_text(body, encoding="utf-8")
print(OUT)
