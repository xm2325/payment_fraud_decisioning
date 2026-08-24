# Moniepoint application notes — v1.8

Use these results only as **portfolio / benchmark evidence**. The 120k stream and PaySim are synthetic; do not present any number as Moniepoint production impact, prevented loss or real analyst performance.

## Recommended CV bullets

Use at most **two** bullets on a one-page CV. The strongest pairing remains the stage-separated full-PaySim routing bullet plus the emerging-fraud bullet. The v1.8 recalibration result is strongest as an interview / system-governance example rather than a third CV bullet.

### Option A — full PaySim + future-safe decision policy

Built and validated a time-aware fraud decisioning pipeline on the full **6.36M-row PaySim** benchmark using DuckDB point-in-time SQL and GitHub Actions; separated model training, probability calibration, routing-policy selection and untouched future evaluation into ordered temporal stages, with a validation-selected `P(fraud) × amount^0.25` review policy reaching **61.6% precision, 23.0% fraud-case recall and 77.7% fraud-value recall at 50 reviews per 10k transactions**.

### Option B — emerging fraud + Fraud Ops

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

### Option C — feedback + label maturity

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the controlled attack design.

## Strong v1.8 interview story

### 1. Prediction, calibration and decision-policy selection are separate lifecycle decisions

The full PaySim evidence path is:

- model training: steps **1--445**;
- initial probability calibration only: **446--519**;
- routing-policy selection only: **520--594**;
- untouched future evaluation: **595--743**.

The validation-stage cutoff is determined from time only, not fraud labels or performance. The predictive feature family is fixed before the calibration/routing experiments.

### 2. Routing remained stable after the stricter split

The routing family is pre-specified as

`priority = P(fraud) × (amount / policy_median_amount)^alpha`, `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`.

Only later validation steps select alpha. Case-first, balanced and value-first objectives all still select **alpha=0.25**. On untouched future PaySim, at exact **50 reviews per 10,000 transactions**, the frozen policy reaches **61.59% precision / 22.97% fraud-case recall / 77.67% fraud-value recall**.

Good phrasing: “I separated calibration from policy selection to check whether the routing compromise was an artefact of validation reuse. It was not: alpha=0.25 survived the stricter temporal protocol.”

### 3. One-time probability calibration drifted

The initial calibration stage has **1.120%** fraud prevalence and **1.125%** mean predicted risk. With the calibrator frozen, later policy-selection risk is over-predicted (**1.662% predicted vs 0.494% observed**) and future risk is also over-predicted (**2.533% vs 1.338%**).

This motivated v1.8: test recalibration using only labels that would have matured as of each future window.

### 4. Mature labels do not justify automatic recalibration

The model, `alpha=0.25` routing policy and review capacity remain frozen. Only the sigmoid calibrator is refreshed.

| Method | Mean 3-window Brier | Worst Brier | Mean abs log risk-ratio error |
|---|---:|---:|---:|
| frozen initial | 0.01578 | **0.02561** | 0.6441 |
| **24h as-of refresh** | **0.01519** | 0.02849 | **0.3868** |
| 7-day as-of refresh | 0.01606 | 0.02647 | 0.7496 |
| instant-history diagnostic | 0.01540 | 0.02905 | 0.5076 |

The 24h as-of refresh substantially improves the first two future windows. In window 1 it moves mean predicted risk from **2.388% to 1.304%** against **1.301% observed prevalence**, and Brier improves from **0.01215 to 0.01043**. In window 2 Brier improves from **0.00957 to 0.00667**.

But the third future window is an abrupt high-fraud regime: observed prevalence jumps to **3.863%**. The 24h calibrator predicts only **1.603%** and Brier becomes **0.02849**, worse than the frozen calibrator's **0.02561**. Even the instant-history diagnostic under-predicts the new regime and has worse Brier.

Good phrasing: “Newer labels improved average calibration but made the abrupt high-fraud period worse. I would not auto-promote a new calibrator just because more labels matured; I would require a mature-label champion/challenger gate.”

### 5. Calibration quality and Fraud Ops queue quality are different contracts

At fixed `alpha=0.25` and 50 reviews/10k, recalibration leaves the selected queue unchanged in future windows 1 and 3. Only window 2 changes modestly: 24h refresh raises case recall from **25.64% to 26.92%** and fraud-value recall from **81.74% to 82.21%**.

This is useful evidence that absolute probability calibration can move materially while exact-capacity ranking remains comparatively stable.

### 6. Capacity saturation is not model failure

At 50 reviews/10k, future fraud-value recall remains roughly **75.45% → 81.74% → 40.76%** under the frozen routing policy. In the final high-fraud window every admitted case is fraud, yet case recall is only **12.77%**.

A perfect-precision queue can still miss most fraud when fraud arrival rate exceeds analyst capacity.

### 7. Keep negative recipient evidence

PaySim has no confirmed mule-account label. Standalone recipient fan-in / recipient-intensity signals have future AUC around **0.47--0.49**, and validation-selected thresholds recover **0% future fraud**.

Good phrasing: “I tested intuitive mule-style recipient signals independently and they did not generalise, so I kept them as investigation diagnostics rather than inventing a mule-detection claim.”

## Controlled 120k evidence worth discussing

- Point-in-time behavioural history raises known-fraud PR-AUC **0.597 → 0.640** and fraud-value recall **78.1% → 86.1%** in the controlled stream.
- An unseen future-only attack has **0% supervised recall** while the label-free tail detector reaches **93.3% recall at 0.86% legitimate flag rate**.
- At 200 reviews/10k, a governance-fixed 80/20 exploit-explore split gives **80.9% value recall / 40.2% novel recall**, versus **81.8% / 0%** for model-only routing.
- A 7-day mature-label view has **0% novel recall** where an invalid instant-label oracle reaches **88.4%**, demonstrating temporal label leakage.
- Investigation-driven labels create verification bias: known-fraud PR-AUC is **0.584** with risk-triggered labels versus **0.646** with full historical labels.

## Claims not to make

Do not claim:

- real Moniepoint data;
- real prevented or saved loss;
- a real A/B treatment effect;
- Moniepoint fraud prevalence, traffic, label maturity or analyst capacity;
- measured production latency/SLA;
- a real analyst-confirmation rate;
- confirmed mule-account detection from PaySim;
- simulator balance-field PR-AUC (~0.995) as a realistic fraud headline;
- that 24-hour recalibration is universally better or is a recommended Moniepoint refresh cadence.

PaySim may be described only as a **public synthetic external benchmark** used to demonstrate scalable point-in-time feature engineering, temporal evaluation, calibration governance and Fraud Ops routing.

## Short interview summary

“I treated fraud detection as a temporal decisioning and operations problem rather than an AUC exercise. On the full 6.36M-row PaySim benchmark, I separated model training, calibration, routing-policy selection and future evaluation. The routing compromise `P(fraud) × amount^0.25` survived the stricter validation-only selection and at 50 reviews per 10k reached 61.6% precision, 23.0% fraud-case recall and 77.7% fraud-value recall. I then tested as-of recalibration under delayed labels: a 24h refresh improved average calibration but made the abrupt high-fraud window worse, while the capacity-constrained review queue barely changed. That led to a governance conclusion: calibration refresh and analyst routing should have separate monitoring and champion/challenger approval gates.”
