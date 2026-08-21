# Moniepoint application notes — v0.9

Use only results clearly labelled as coming from the **120,000-transaction synthetic payment-stream case study**. Do not present any number below as Moniepoint production impact.

## Strong CV bullets

**Option A — modelling + decisioning**

Built a time-aware payment-fraud decisioning case study on **120,000 synthetic transactions**, with point-in-time Python/SQL features, calibrated LightGBM and loss/friction policies; behavioural history increased known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at a validation-derived ~1% legitimate-flag operating point.

**Option B — emerging fraud + Fraud Ops**

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

**Option C — feedback + label maturity**

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Use at most two of these bullets on the CV. Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the synthetic attack design.

## Interview findings worth discussing

### Feature selection is empirical

Adding network-style history to the supervised model lowered known-fraud PR-AUC from **0.640 to 0.635**. The project therefore keeps those signals in anomaly/investigation rather than claiming every fraud-specific feature belongs in the production classifier.

### Fraud Ops has two learning goals

Exploitation catches known high-loss fraud. Exploration finds behaviour the supervised model does not understand. At 200 reviews/10k, an 80/20 split gives **80.9% value recall and 40.2% novel recall**, versus **81.8% and 0%** for model-only routing.

### Anomaly detection should create labels

The anomaly model is most useful when it shortens the time to obtain confirmed examples. In the controlled feedback experiment, 10 confirmed emerging cases raise later novel recall to **38.0%**; 100 reviews yielding 95 novel cases raise it to **89.3%**.

### Verification bias matters

A stress test in which labels mainly come from risk-triggered follow-up produces a training sample with **8.24% fraud prevalence** even though the full historical population is **1.13%**. It yields known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels. A random audit lane improves coverage, but the repo does not claim that 10% is an optimal real-world audit rate.

### Policy numbers depend on business assumptions

Across four stated efficacy/cost scenarios, the validation-selected block threshold remains **0.25**, while the review threshold ranges **0.03-0.06**. Future fraud-value prevention ranges **65.6-84.3%** and legitimate friction **2.72-4.88%**. Use this to explain why the reference 78.5% is a scenario result rather than a business claim.

### Precision is base-rate dependent

At the measured supervised TPR/FPR, expected precision is only **3.0%** if deployment fraud prevalence is 0.1%, and **23.8%** if prevalence is 1%. This is why synthetic precision should not be used directly for staffing or customer-friction estimates.


### Probability calibration must be market-aware

The validation calibrator is fitted at about 1.07% fraud prevalence. Under a controlled 0.10% prior-shift simulation, the unadjusted model averages **0.679% predicted risk** while the adjusted posterior averages **0.085%**, with Brier improving from **0.001665 to 0.000855**. Do not oversell this: at 2% prevalence the correction slightly worsens Brier because the future period also contains concept drift. The interview point is that base-rate correction is conditional on a label-shift assumption and must be monitored.

### A good detector can still overwhelm Fraud Ops

The reference review + exploration policy creates about **5.42 candidate cases/hour**. In the queue stress test, 4 reviews/hour is structurally overloaded at 1x traffic; 6 reviews/hour stays below average load and meets a four-hour wait proxy in the reference stream, but a 1.5x traffic increase overloads even 8 reviews/hour. These are scenario capacities, not recommendations for Moniepoint. Use this to discuss queue health, service-level guardrails and why threshold changes may be required during traffic spikes.

### SQL correctness is tested, not decorative

The repo executes a SQLite point-in-time reference query and verifies parity with the Python feature builder, including equal timestamps. This is useful preparation for the SQL/Python take-home stage.

## Claims not to make

Do not claim: real Moniepoint data, real prevented loss, a real A/B treatment effect, production fraud prevalence, a measured production latency/SLA, or a real analyst-confirmation rate. PaySim results may be described only as a **public synthetic external benchmark**, never as production performance.

The full PaySim benchmark is now verified through GitHub Actions. The runner downloads the canonical 6,362,620-row dataset, performs the count audit, computes DuckDB point-in-time features and uploads aggregate results only.

### v1.0 operational-control bullet

- Built a backlog-aware two-lane fraud-review controller that converts model/anomaly alerts into a capacity-feasible analyst queue. In the 120k synthetic stress test, a fixed 6-review/hour budget accepted **88.3%** of candidates at 1x traffic while preserving **94.0%** system fraud-value coverage; at 1.5x traffic, acceptance fell to **67.8%** and novel-fraud recall to **60.3%**, quantifying the detection cost of protecting review capacity.

Use this bullet only with the words **synthetic stress test** nearby. The 6-review/hour capacity and traffic multipliers are scenario settings, not Moniepoint staffing or traffic estimates.


## Full PaySim external validation — safe application wording

Preferred evidence: ran a reproducible GitHub Actions benchmark on the complete **6.36M-row PaySim** mobile-money simulator using DuckDB point-in-time SQL and a strict future split. The balance-free validation-only model selection chose relational/pair/counterparty features (validation PR-AUC **0.3252** vs **0.2775** transaction-only). On the untouched future test, the selected model reached PR-AUC **0.3530**, **60.1% precision**, **25.9% fraud recall**, **80.7% fraud-value recall** and **0.233% legitimate flags**. A validation-thresholded amount/type rule reached only **46.6% precision, 12.1% recall and 62.2% value recall**.

A separate audit found that PaySim old/new-balance derivatives lift PR-AUC to **0.996**. Do **not** use 0.996 as a CV headline. The correct interpretation is that simulator balance mechanics are unusually informative, which is why the portfolio reports a balance-free benchmark separately.

Possible CV bullet:

> Built and validated a time-aware payment-fraud pipeline on the full 6.36M-row PaySim benchmark using DuckDB point-in-time SQL and GitHub Actions; the balance-free model reached PR-AUC 0.353 with relational features and captured 80.7% of fraud value; against a validation-thresholded amount/type rule it more than doubled fraud-case recall (12.1% to 25.9%) while raising precision from 46.6% to 60.1%, while a feature audit identified simulator balance fields as a source of unrealistically high separability.
