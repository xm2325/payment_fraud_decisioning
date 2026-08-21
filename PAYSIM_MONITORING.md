# PaySim capacity and monitoring contract — v1.4

This document defines the operational monitoring contract for the full **6,362,620-row** PaySim benchmark. It deliberately separates **ranking quality**, **review-capacity routing**, and **retrospective threshold diagnostics**.

## 1. Time and model contract

The balance-free reference model is `transaction_plus_relational`.

1. Train on steps 1--445.
2. Calibrate and select the balance-free model using validation steps 446--594 only.
3. Lock the model before seeing future steps 595--743.
4. Evaluate future ranking and operational routing without using future labels to choose the model or review capacity.

PaySim records time as hourly `step`, so all relational/history features use only prior steps. Same-step transactions are treated as simultaneous.

## 2. Why a scalar threshold is diagnostic, not the capacity contract

The original benchmark used a validation quantile threshold to target a narrow legitimate-alert rate. The v1.4 audit exposed an important edge case: LightGBM produces large score ties, so a rule of `score >= threshold` can either exceed a narrow budget or, after a tie-safe hard-cap fix, materially under-use it.

For the selected relational model, a tie-safe target of **0.1% validation legitimate flags** locks a threshold near **0.246869** but actually uses only **0.00221%** legitimate flags on validation. On the future test it generates **291 alerts**, all fraud, with **17.59% fraud-case recall** and **44.18% fraud-value recall**. This is useful as a threshold-behaviour diagnostic, but it is not a sensible way to allocate a fixed analyst queue.

`threshold_budget_drift.csv`, `future_threshold_windows.csv` and `future_posthoc_threshold_cap.csv` retain this diagnostic evidence. The post-hoc future cap is explicitly retrospective and must not be described as a deployable result.

## 3. Operational contract: exact ranked capacity

The operational contract is therefore **total alerts per 10,000 transactions**, independent of labels. Within each evaluation period, cases are ranked by a non-label score and the top `k` are admitted. Equal scores are broken only by a stable `event_key` derived from transaction fields; `event_key` is never a model feature.

For calibrated model probability ranking on the untouched future test:

| Alerts / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | **100.0%** | **7.44%** | 32.62% |
| 25 | **96.75%** | **18.02%** | 46.39% |
| 50 | **64.34%** | **24.00%** | 71.67% |
| 100 | 41.05% | 30.65% | 83.33% |

These are capacity-constrained benchmark results, not staffing recommendations for Moniepoint.

## 4. Three routing objectives at identical capacity

v1.4 compares three rankers with exactly the same review slots:

- `relational_model_probability`: prioritise probability of fraud;
- `model_probability_x_amount`: prioritise a simple expected-loss heuristic `P(fraud) × amount`;
- `amount_type_rule`: prioritise `TRANSFER` / `CASH_OUT` by amount as an interpretable baseline.

| Alerts / 10k | Ranker | Precision | Fraud-case recall | Fraud-value recall |
|---:|---|---:|---:|---:|
| 10 | probability | **100.0%** | **7.44%** | 32.62% |
| 10 | probability × amount | 95.12% | 7.07% | 33.22% |
| 10 | amount/type rule | 74.80% | 5.56% | **33.79%** |
| 25 | probability | **96.75%** | **18.02%** | 46.39% |
| 25 | probability × amount | 69.16% | 12.88% | **61.45%** |
| 25 | amount/type rule | 53.57% | 9.98% | 54.90% |
| 50 | probability | **64.34%** | **24.00%** | 71.67% |
| 50 | probability × amount | 56.40% | 21.04% | **76.96%** |
| 50 | amount/type rule | 42.46% | 15.84% | 70.43% |
| 100 | probability | 41.05% | 30.65% | 83.33% |
| 100 | probability × amount | **42.51%** | **31.74%** | **87.37%** |
| 100 | amount/type rule | 34.41% | 25.70% | 83.03% |

The conclusion is deliberately not “one ranker wins”. Probability ranking is strongest for clean case capture at tight/medium capacity; `P(fraud) × amount` trades some case coverage for higher value coverage at 25--50 alerts/10k and dominates the probability ranker at 100/10k in this future period. The amount/type rule remains competitive for value when the queue is extremely tight because fraud value is highly concentrated.

`P(fraud) × amount` is only an expected-loss **prioritisation heuristic**. It is not prevented loss and inherits probability-calibration, amount-quality and intervention-efficacy assumptions.

## 5. Future-window capacity stress

At a fixed **50 alerts / 10k**, the model-probability queue has fraud-value recall **74.34%**, **80.99%**, then **40.76%** across future windows 595--644, 645--694 and 695--743. The last window has fraud prevalence **3.86%**; all 71 admitted model-ranked cases are fraud, yet case recall is only **12.77%**. That is a capacity-saturation failure, not a false-positive problem.

In that last high-fraud window, all three rankers admit 71 fraud cases and therefore have the same **100% precision / 12.77% case recall**, but value recall differs materially:

- probability: **40.76%**;
- probability × amount: **61.97%**;
- amount/type rule: **64.78%**.

This is the operational reason to monitor queue saturation and value mix separately from classifier ranking metrics.

## 6. Recipient investigation signals: negative evidence

PaySim does not contain a confirmed mule-account label, so the recipient audit does **not** claim mule detection. Strict prior-step recipient signals are retained as an investigation audit only.

On the future test, `recipient_fanin_24h` has AUC about **0.493** and the composite recipient-intensity score about **0.467**. Validation-selected standalone recipient thresholds recover **0% future fraud**. The project therefore does not promote recipient activity as a defensible mule proxy in PaySim.

Negative evidence is kept because it demonstrates that investigation features are tested rather than added for narrative fit.

## 7. Reproducibility boundary

The materialised PaySim table now carries a stable non-label `event_key` and split loading uses deterministic `ORDER BY step, event_key`. Independent full-benchmark and monitoring workflows reproduce the selected relational model and locked scalar threshold to numerical precision.

The GitHub Actions monitoring workflow validates the canonical PaySim counts before running and uploads only aggregate outputs. Raw transactions, materialised features and fitted models remain runner-local.

## 8. Evidence boundary

PaySim is synthetic mobile-money data. These results demonstrate temporal evaluation, point-in-time feature engineering, ranking, capacity routing and monitoring design. They are **not** production fraud rates, real customer-friction estimates, confirmed mule-account findings, real staffing requirements or saved-money claims.
