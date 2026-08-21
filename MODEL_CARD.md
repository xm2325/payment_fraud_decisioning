# Model card — v0.9

## Intended use

Portfolio demonstration of payment-fraud modelling and decision-system design: supervised risk scoring, label-free anomaly detection, analyst-capacity routing, temporal evaluation, label latency, analyst feedback and monitoring.

## Not intended for

Real payment blocking, customer challenge decisions, loss forecasting, fraud-rate estimation, or claims about Moniepoint. The default dataset is synthetic.

## Supervised champion

Calibrated LightGBM trained on static transaction features plus strictly backward-looking velocity/history. Network-style history is excluded from the champion because the v0.9 reference ablation shows no improvement on known-fraud PR-AUC.

## Early-warning channel

An interpretable upper-tail detector is trained only on historical legitimate transactions. It is used as an investigation signal, not as proof that a transaction is fraudulent.

## Operational policy

Validation-derived risk thresholds map transactions to approve/review/block. A separate fixed-capacity exploit-explore review queue reserves analyst capacity for anomaly discovery. Intervention efficacy and costs are stated simulation assumptions.

## Label timing

Historical fraud outcomes are treated as unavailable until mature. The delayed-label experiment shows that using instant future labels materially inflates retraining performance.

## Analyst feedback

The feedback experiment assumes anomaly-ranked cases can receive expedited analyst confirmation during a discovery window. Those confirmed outcomes are then added to supervised training before a later future evaluation period. This is a method demonstration; real investigation latency and confirmation quality are unknown.

## Verification bias

The verification-bias stress test uses a synthetic follow-up rule and random audit lane. Its purpose is to show that investigation policy can shape training-label coverage and typology recall. The specified audit percentages are not recommendations for a real fraud team.

## Policy-assumption sensitivity

The approve/review/block impact depends on intervention efficacy and friction/case-cost assumptions. v0.9 reports four pre-specified scenarios rather than treating one cost matrix as known. The reference prevented-value result must always be presented with this limitation.

## Base-rate limitation

Precision is prevalence-dependent. v0.9 reports Bayes-adjusted precision sensitivity from 0.1% to 2% fraud prevalence so synthetic observed precision is not mistaken for a production expectation.


## Prior-shift calibration

A prior-probability correction is included as a sensitivity analysis for markets or products with different fraud prevalence. It assumes stable class-conditional score distributions. The 120k stress test shows strong improvement at low target prevalence but a small Brier deterioration at 2%, where the future period also contains concept drift. Production use would require ongoing calibration checks and segment-specific base rates.

## Queue-capacity limitation

Model thresholds create operational arrivals. v0.9 therefore simulates hourly review/exploration arrivals against fixed analyst capacities and reports utilisation, backlog and a wait-time proxy. Staffing levels and the four-hour threshold are hypothetical scenarios and must not be interpreted as Moniepoint operating requirements.

## Main 120k reference results

- Known-fraud PR-AUC: 0.597 static -> **0.640** with velocity/history.
- Known-fraud value recall: 78.1% -> **86.1%** at the validation-derived ~1% legitimate-flag operating point.
- Test-only attack: supervised **0% recall**; tail detector **93.3% recall at 0.86% legitimate flag rate**.
- Fixed 80/20 queue at 200 reviews/10k: **80.9% fraud-value recall + 40.2% novel recall**.
- 100 anomaly-ranked analyst reviews in the feedback simulation: **95 confirmed novel cases; 89.3% later novel recall**.
- 7-day matured-label view: **0%** later novel recall vs **88.4%** for an invalid instant-label oracle.

## Required real-world additions

Confirmed fraud/chargeback timestamps, label maturity rules, intervention assignments, review outcomes, customer challenge completion, device fingerprint quality, account/merchant network history, operational queue capacity, cost estimates and monitored production base rates.

## v1.0 operational note

The model score is not itself the analyst workload. v1.0 adds a separate capacity-aware admission controller. Under the 120k synthetic reference stream and a scenario capacity of 6 reviews/hour, it keeps scaled review utilisation below 100% across 1x-4x traffic by raising effective admission cutoffs. This protects queue stability but reduces novel-fraud recall from 81.2% at 1x to 20.5% at 4x. These are stress-test results, not production staffing requirements.


## PaySim external validation

The repository runs the complete 6,362,620-row PaySim synthetic mobile-money dataset through a separate DuckDB/LightGBM temporal benchmark. The balance-free ablation shows transaction-only PR-AUC **0.3403**; adding basic prior-step history gives **0.3408**; adding relational pair/counterparty history gives **0.3530**. At their validation-derived operating points, the relational model has **60.1% precision, 25.9% fraud recall, 80.66% fraud-value recall and 0.233% future legitimate flag rate**. Relative to transaction-only, it improves ranking/selectivity but slightly lowers fraud coverage.

The model with PaySim balance derivatives reaches PR-AUC **0.9950** and near-100% recall. Because these variables reflect simulator accounting mechanics, this is recorded as a dataset-sensitivity finding and is not used as the deployment-style performance estimate.

The final v1.3 code selects its balance-free reference model using **validation PR-AUC only** and then freezes that choice for future-test evaluation. This avoids choosing transaction/history/relational features by looking at test performance. PaySim remains synthetic and no result in this section is a Moniepoint or production estimate.


### PaySim simple-rule baseline and value concentration

A validation-thresholded `TRANSFER/CASH_OUT + amount` rule reaches 46.6% precision, 12.1% fraud recall and 62.2% fraud-value recall on the future test. The selected relational model reaches 60.1%, 25.9% and 80.7%. Because the top 25% of PaySim fraud cases account for 82.4% of fraud value, fraud-value recall is not interpreted as a stand-alone model-quality metric.
