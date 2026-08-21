# Moniepoint application notes — v1.4

Use these results only as **portfolio / benchmark evidence**. The 120k stream and PaySim are synthetic; do not present any number as Moniepoint production impact, prevented loss or real analyst performance.

## Recommended CV bullets

Use at most **two** bullets on a one-page CV. The strongest pairing for a fraud Data Scientist application is the full PaySim capacity bullet plus the emerging-fraud bullet.

### Option A — full PaySim + Fraud Ops capacity

Built and validated a time-aware fraud pipeline on the full **6.36M-row PaySim** benchmark using DuckDB point-in-time SQL and GitHub Actions; validation-only model selection chose relational features (**0.323 validation / 0.350 future PR-AUC**), and at a fixed **50 reviews per 10k transactions** the model increased fraud-case recall from **15.8% to 24.0%** and precision from **42.5% to 64.3%** versus an amount/type rule.

This is the preferred PaySim CV bullet because model and baseline use **exactly the same analyst capacity**.

### Option B — emerging fraud + Fraud Ops

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

### Option C — feedback + label maturity

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the controlled attack design.

### Optional operations bullet

Built a backlog-aware two-lane fraud-review controller that converts model/anomaly alerts into a capacity-feasible analyst queue; in a **120k synthetic stress test**, a fixed 6-review/hour scenario accepted **88.3%** of candidates at 1x traffic, while 1.5x traffic reduced acceptance to **67.8%** and novel-fraud recall to **60.3%**, quantifying the discovery cost of protecting review capacity.

The 6-review/hour setting and traffic multipliers are scenario assumptions, not Moniepoint staffing estimates.

## Full PaySim evidence to discuss in interview

### 1. Model selection is genuinely future-safe

The full workflow verifies **6,362,620 transactions / 8,213 fraud cases / steps 1--743**, computes strict prior-step features in DuckDB, then uses train 1--445, validation 446--594 and future test 595--743.

Balance-free validation PR-AUC:

- transaction only: **0.2790**;
- transaction + history: **0.2774**;
- transaction + relational: **0.3228**.

The relational model is selected before the future period is evaluated and reaches future PR-AUC **0.3497**.

### 2. Do not use the old v1.3 operating-point numbers

Earlier notes used **60.1% precision / 25.9% recall / 80.7% fraud-value recall** from a narrow validation quantile threshold. v1.4 found that large score ties made that alert-budget contract unsafe.

Those values are **superseded** and should not appear in CV, cover letter or interview answers.

The corrected scalar threshold is retained only as a diagnostic; exact analyst capacity is the operational contract.

### 3. Compare model and rule at identical review capacity

At **50 alerts per 10,000 transactions** on the untouched future test:

| Ranker | Precision | Fraud recall | Fraud-value recall |
|---|---:|---:|---:|
| relational model probability | **64.34%** | **24.00%** | 71.67% |
| probability × amount | 56.40% | 21.04% | **76.96%** |
| amount/type rule | 42.46% | 15.84% | 70.43% |

The strongest modelling claim is therefore **case capture and precision at the same review cost**, not “ML catches 80% of fraud value”.

### 4. Fraud probability and expected loss are different objectives

At 50/10k, `P(fraud) × amount` gives up **2.96 percentage points** of case recall versus probability ranking but gains **5.28 points** of fraud-value recall.

At 100/10k, expected-loss ranking reaches **42.51% precision / 31.74% recall / 87.37% value recall**, slightly ahead of pure probability ranking on all three metrics in that future period.

Good interview phrasing: “I would not choose the queue score from model AUC alone. Fraud Ops needs an explicit objective: maximise confirmed case yield, expected loss coverage, customer experience, or a governed combination.”

Do not call `P(fraud) × amount` prevented loss. It is a prioritisation heuristic and assumes the model probabilities and transaction amounts are meaningful for ranking.

### 5. Capacity saturation can look like model failure

At a fixed 50/10k, probability-ranked fraud-value recall across the three future windows is **74.3% → 81.0% → 40.8%**. In the last window, fraud prevalence rises to **3.86%** and all 71 reviewed cases are fraud, but case recall is only **12.8%**.

That is a useful interview example: precision can be 100% while the operation still misses most fraud because demand exceeds analyst capacity.

### 6. Negative recipient result is worth keeping

PaySim does not have a confirmed mule-account label. Standalone recipient fan-in / recipient-intensity signals have future AUC around **0.47--0.49**, and validation-selected thresholds recover **0% future fraud**.

Good interview phrasing: “I tested the intuitive mule-style recipient signals independently and they did not generalise, so I kept them as investigation evidence rather than inventing a mule-detection claim.”

## Controlled 120k findings worth discussing

### Feature selection is empirical

Adding network-style history to the supervised model lowers known-fraud PR-AUC from **0.640 to 0.635**. Those signals therefore stay in anomaly/investigation rather than being forced into the supervised champion.

### Fraud Ops has two learning goals

At 200 reviews/10k, model-only routing gives **81.8% value recall / 0% novel recall**. A governance-fixed 80/20 exploit-explore split gives **80.9% / 40.2%**.

### Anomaly detection should create labels

In the controlled feedback experiment, 10 confirmed emerging cases raise later novel recall to **38.0%**; 100 reviews yielding 95 simulated novel cases raise it to **89.3%**.

### Verification bias matters

Risk-triggered follow-up produces a labelled training sample with **8.24% fraud prevalence** even though the full historical population is **1.13%**. Known-fraud PR-AUC is **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full labels.

### Policy numbers depend on assumptions

Across four stated efficacy/cost scenarios, future fraud-value prevention ranges **65.6--84.3%** and legitimate friction **2.72--4.88%**. The reference 78.5% is therefore a simulator scenario result, not a business claim.

### Precision is base-rate dependent

Holding measured TPR/FPR fixed, expected alert precision is only about **3.0%** at 0.1% fraud prevalence and **23.8%** at 1%. Synthetic observed precision should not be used directly for production staffing.

### SQL correctness is tested, not decorative

The repository executes a SQLite point-in-time reference query and checks Python parity, including equal timestamps. The full PaySim workflow separately uses strict prior-step DuckDB windows and deterministic non-label event keys.

## Claims not to make

Do not claim:

- real Moniepoint data;
- real prevented or saved loss;
- a real A/B treatment effect;
- Moniepoint fraud prevalence, traffic or analyst capacity;
- a measured production latency/SLA;
- a real analyst-confirmation rate;
- confirmed mule-account detection from PaySim;
- simulator balance-field PR-AUC (~0.995) as a realistic fraud headline.

PaySim may be described as a **public synthetic external benchmark** used to demonstrate scalable point-in-time feature engineering, future-safe evaluation and Fraud Ops routing.

## Short interview summary

“I treated fraud detection as a decisioning and operations problem rather than an AUC exercise. On the full 6.36M-row PaySim benchmark I selected the balance-free relational model only on validation data, then compared model probability, expected-loss and an interpretable amount rule at identical analyst capacities. At 50 reviews per 10k, model probability raised precision from 42.5% to 64.3% and case recall from 15.8% to 24.0% versus the amount rule; ranking by probability times amount increased value recall further to 77.0%, but with lower case recall. The project also stress-tests unseen fraud, delayed labels, verification bias and queue saturation, and keeps negative recipient-signal results rather than claiming unsupported mule detection.”
