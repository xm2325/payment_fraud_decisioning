# Result snapshot — v1.3

## Full PaySim external-validation result

A GitHub-hosted full run verified the canonical **6,362,620-row / 8,213-fraud / step-1-to-743** PaySim table and evaluated a strict future split (train 1--445, validation 446--594, test 595--743). Fraud prevalence rises from **0.0833%** in train to **0.6804%** in validation and **1.3384%** in test.

At thresholds selected on validation, the balance-free ablation is:

| model | PR-AUC | precision | recall | future legit flag | fraud-value recall |
|---|---:|---:|---:|---:|---:|
| transaction only | 0.3403 | 51.0% | 28.2% | 0.367% | 81.16% |
| + prior-step history | 0.3408 | 54.4% | 27.6% | 0.314% | 81.15% |
| + relational history | **0.3530** | **60.1%** | 25.9% | **0.233%** | 80.66% |
| + simulator balances | 0.9950 | 87.1% | 99.94% | 0.201% | 99.99% |

The relational model is selected **before test evaluation** because its validation PR-AUC is **0.3252**, versus **0.2775** for transaction-only and **0.2733** for basic history. Relational history improves PR-AUC and selectivity but slightly reduces coverage at this operating point. The simulator-balance row is retained only as a dataset-mechanics sensitivity. PaySim's source `isFlaggedFraud` rule produces 8 future alerts, with 100% precision but **0.48% recall / 1.56% fraud-value recall**.


A validation-thresholded `TRANSFER/CASH_OUT + amount` rule reaches **46.6% precision, 12.1% fraud recall, 62.2% fraud-value recall and 0.188% legitimate flags** on the future test. The selected relational model reaches **60.1%, 25.9%, 80.7% and 0.233%** respectively. Fraud value is concentrated: the largest **10% / 25% / 50%** of fraud cases account for **55.1% / 82.4% / 95.3%** of fraud value, so value recall is never interpreted without case recall and the amount-rule baseline.

The relational full-data run takes about **216 seconds** for DuckDB feature materialisation plus four LightGBM/calibration fits after data download. Raw PaySim data are not committed. Aggregate evidence is stored under `results/paysim_full/`.

## 1. Supervised feature ablation

Known-fraud evaluation at a validation-derived operating point of about 1% legitimate flags:

| Feature set | PR-AUC | Fraud recall | Fraud-value recall |
|---|---:|---:|---:|
| Static transaction features | 0.597 | 60.2% | 78.1% |
| + point-in-time velocity/history | **0.640** | **66.9%** | **86.1%** |
| + network-style history | 0.635 | 64.3% | 84.1% |

Network-style history does not improve the supervised champion in this run. It remains available for anomaly detection and investigation reason codes.

## 2. Decision policy

Review/block thresholds are chosen on validation, not future test data. Under the stated simulated intervention-efficacy assumptions, the chosen policy prevents **78.5%** of test fraud value while **4.88%** of legitimate transactions enter review or block. The complete frontier is in `policy_frontier_test.csv`.

## 3. Unseen-attack stress test

| Detector | Known-fraud recall | New-attack recall | Legitimate flag rate |
|---|---:|---:|---:|
| Supervised | 66.9% | **0.0%** | 1.15% |
| Isolation Forest | 2.6% | 31.8% | 0.90% |
| Interpretable tail detector | 0.4% | **93.3%** | 0.86% |
| Supervised OR tail | 66.9% | **93.3%** | 2.00% |

The anomaly channel succeeds because the simulator's new attack was designed to be abnormal in shared-device/velocity history while looking ordinary on common supervised cues. This is a controlled stress test, not a real-world discovery-rate estimate.

## 4. Fraud Ops review capacity

At **100 reviews / 10k**, model-only routing captures **66.5%** of fraud value and **0%** of the new attack. Fixed 80/20 exploit-explore routing captures **62.8%** of fraud value and **20.1%** of the new attack.

At **200 reviews / 10k**, model-only gets **81.8% value recall / 0% new-attack recall**; fixed 80/20 gets **80.9% / 40.2%**. The 80/20 split is governance-defined, not selected using future novel-fraud labels.

## 5. Analyst feedback closes the loop

The anomaly channel ranks days 48-53 for expedited analyst review. Confirmed results are added to historical training, then the model is evaluated on later days 54-59.

| Reviews | Confirmed novel cases | Future novel recall | Future fraud-value recall |
|---:|---:|---:|---:|
| 0 | 0 | 0.0% | 80.0% |
| 10 | 10 | 38.0% | 82.9% |
| 50 | 50 | 56.2% | 83.3% |
| 100 | 95 | **89.3%** | 85.7% |
| 200 | 107 | 91.7% | 85.7% |
| 500 | 108 | 90.9% | 85.1% |

The top anomaly queue is intentionally rich in the simulated attack, so this curve is an upper-bound-style method demonstration. It shows the intended operational loop: anomaly detection -> analyst confirmation -> supervised learning.

## 6. Label latency

For days 54-59, a **7-day as-of matured-label** training view gets PR-AUC **0.370** and **0% novel recall**. An invalid instant-label oracle gets PR-AUC **0.817** and **88.4% novel recall**. The oracle is not deployable; it quantifies the optimism caused by assuming future labels are already known.

## 7. Verification bias and random audit coverage

A synthetic historical follow-up rule labels transactions with device change, country mismatch or top-3% amount; a random audit samples outside that set. The validation period is treated as independently audited so thresholding remains comparable.

With **risk-triggered labels only**, the labelled training set is 7.73% of historical transactions and has fraud prevalence 8.24%, versus 1.13% in the full historical population. Known-fraud PR-AUC is **0.584**; mule-cashout recall is **11.1%**. With full historical labels these become **0.646** and **28.6%**.

Across two fixed random seeds, adding a **10% random audit** outside triggered cases gives mean known-fraud PR-AUC **0.618** and mule-cashout recall **20.6%**. The point is not that 10% is optimal; it is that label collection policy changes what the model can learn.

## 8. Base-rate sensitivity

Using Bayes' rule while holding measured TPR/FPR fixed:

| Assumed fraud prevalence | Expected supervised alert precision | Expected tail-alert precision |
|---:|---:|---:|
| 0.10% | **3.0%** | **4.9%** |
| 0.25% | 7.2% | 11.4% |
| 0.50% | 13.5% | 20.5% |
| 1.00% | 23.8% | 34.1% |
| 2.00% | 38.7% | 51.2% |

This does not assume the real Moniepoint fraud rate. It demonstrates why observed precision from a synthetic stream with ~2.1% test fraud prevalence cannot be transferred to a lower-base-rate production setting.

## 9. Policy-assumption sensitivity

The reference policy is not treated as a universal optimum. Four pre-specified scenarios vary review/block efficacy and operational/customer-friction cost units. Validation selects the policy independently in each scenario.

| Scenario | Review threshold | Block threshold | Test fraud-value prevented | Test legitimate friction | Test-grid regret |
|---|---:|---:|---:|---:|---:|
| Reference | 0.03 | 0.25 | 78.5% | 4.88% | 0.0% |
| High customer friction | 0.06 | 0.25 | 76.6% | 2.72% | 0.0% |
| Conservative intervention | 0.06 | 0.25 | 65.6% | 2.72% | 0.0% |
| Strong intervention / low friction | 0.03 | 0.25 | 84.3% | 4.88% | 0.0% |

The stable block threshold and zero retrospective grid regret are encouraging within this simulator, but the prevented-value range shows why the reference 78.5% cannot be reported without its assumptions. The scenario costs are not estimates of Moniepoint economics.

## 10. Rolling temporal backtest

Before the new attack, future PR-AUC is **0.661** and **0.571** with fraud-value recall **88.1%** and **86.1%**. Once the attack begins, PR-AUC falls to **0.330-0.374** while fraud-value recall remains **81.3-84.4%** because the missed new attack is low-value. A value-weighted KPI can therefore look acceptable while a new fraud family is being missed.

## 11. Typology sizing and monitoring

Account takeover is **25.2%** of fraud transactions but **52.9%** of fraud value. When the new attack begins, model-score PSI remains close to zero while tail-anomaly alert rate rises from about **0.8-1.0%** to **1.5-1.8%**, moving monitoring status to `INVESTIGATE`.

## 12. Experiment design

The validation review band has **6.23%** fraud incidence. A two-arm randomized intervention test targeting a 25% relative reduction requires about **3,332 transactions per arm** at two-sided alpha 0.05 and 80% power. No customer-completion treatment effect is claimed because the synthetic data do not contain that outcome.

## 13. SQL/Python parity

Equal-timestamp events are processed as a batch in Python and excluded from one another in the SQL reference semantics. `tests/test_sql_parity.py` executes the SQLite query and checks parity for sender 1h/24h velocity, recipient fan-in and device activity.

## 14. PaySim external-validation status

Full external validation is now verified on GitHub Actions. The canonical 6.36M-row dataset is downloaded on the runner, checked against the standard row/fraud/step counts, transformed with strict prior-step DuckDB windows, and evaluated with balance-free and simulator-balance feature sets. The latest verified relational run is GitHub Actions run `32476220879`; only aggregate outputs are retained in the repository.

## 15. Prior-shift calibration stress test

The validation calibrator is fitted at about **1.07% fraud prevalence**. To isolate base-rate shift, the future score distributions are held fixed within class while class prevalence is changed by stratified down-sampling. At an emulated **0.10%** prevalence, the unadjusted posterior has mean predicted risk **0.679%**, Brier **0.001665** and 10-bin ECE **0.00582**. A prior-probability correction gives mean predicted risk **0.085%**, Brier **0.000855** and ECE **0.00022**.

At a **2%** target prevalence, prior correction increases Brier from **0.015209** to **0.015613** even though mean predicted risk moves closer to the new base rate. This is an important failure case: the test period contains a new fraud mechanism, so a pure label-shift assumption is incomplete. The correction is therefore presented as a sensitivity tool, not a production recalibration rule.

## 16. Fraud Ops queue saturation

The reference review-band plus anomaly-exploration policy generates an average **5.42 review candidates/hour** over the 12-day future test window. The queue simulation reserves 20% of hourly service capacity for exploration but lets unused capacity spill between lanes.

| Traffic | Analyst capacity | Utilisation | End backlog | Max wait proxy | 4h proxy met |
|---:|---:|---:|---:|---:|:---:|
| 1.0x | 4/hour | 135.5% | 409 | 102.25 h | No |
| 1.0x | 6/hour | 90.3% | 1 | 3.83 h | Yes |
| 1.0x | 8/hour | 67.8% | 0 | 1.00 h | Yes |
| 1.5x | 8/hour | 101.6% | 44.5 | 7.63 h | No |

The four-hour target, staffing capacities and traffic multipliers are scenario assumptions. The result is not a Moniepoint staffing estimate. Its purpose is to show that alert policy and analyst capacity must be evaluated together: a detector can retain predictive quality while the operational queue becomes unstable.

## v1.0 — backlog-aware capacity admission

v0.9 showed that the fixed review policy can overload Fraud Ops. v1.0 closes that loop: each hour, the system first identifies exploitation candidates from the calibrated model and exploration candidates from the label-free tail detector, then admits only the cases that fit a fixed analyst capacity. A 20% exploration reservation is accumulated across hours so integer rounding cannot starve discovery when only one or two slots are available.

| Traffic | Candidate acceptance | Capacity utilisation after admission | System fraud-value coverage* | Novel-fraud recall* | Legitimate review rate | Median dynamic exploit cutoff |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0x | 88.3% | 79.8% | 94.0% | 81.2% | 4.70% | 0.040 |
| 1.5x | 67.8% | 91.9% | 91.3% | 60.3% | 3.59% | 0.051 |
| 2.0x | 53.2% | 96.2% | 89.7% | 46.9% | 2.78% | 0.069 |
| 4.0x | 27.2% | 98.1% | 83.1% | 20.5% | 1.44% | 0.109 |

*System coverage includes fraud already caught by automatic block plus fraud admitted to timely analyst review. It is a coverage metric, not a prevented-loss claim; review efficacy is not applied here. Traffic multipliers preserve the source-stream score/class mix and reduce the effective number of unique source cases that can fit the fixed hourly capacity. They are stress-test assumptions, not a Moniepoint traffic forecast.

The main operational result is not that dynamic gating "solves" overload. It makes the trade-off explicit: the queue stays capacity-feasible, but the model cutoff rises and emerging-fraud discovery falls as traffic grows. At 4x traffic, the median exploit cutoff increases to 0.109 and novel-fraud recall falls to 20.5%.
