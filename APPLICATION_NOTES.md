# Moniepoint application notes — v1.8

Use these results only as **portfolio / benchmark evidence**. The 120k stream and PaySim are synthetic; do not present any number as Moniepoint production impact, prevented loss or real analyst performance.

## Recommended CV bullets

Use at most **two** bullets on a one-page CV. For a fraud data-science role, the strongest default pairing remains the full-PaySim decision-policy bullet plus the emerging-fraud bullet. The rolling-refresh bullet is a good replacement when the role stresses monitoring, model lifecycle or risk-score calibration.

### Option A — full PaySim + future-safe decision policy

Built and validated a time-aware fraud decisioning pipeline on the full **6.36M-row PaySim** benchmark using DuckDB point-in-time SQL and GitHub Actions; separated model training, probability calibration, routing-policy selection and untouched future evaluation into ordered temporal stages, with a validation-selected `P(fraud) × amount^0.25` review policy reaching **61.6% precision, 23.0% fraud-case recall and 77.7% fraud-value recall at 50 reviews per 10k transactions**.

This remains the preferred general PaySim bullet because the routing policy is selected without reusing calibration-stage labels and all reported future metrics use an exact analyst-capacity contract.

### Option B — emerging fraud + Fraud Ops

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

### Option C — feedback + label maturity

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the controlled attack design.

### Option D — rolling lifecycle + calibration drift

Built a three-cycle rolling-origin fraud model lifecycle on the full **6.36M-row PaySim** benchmark, with expanding-history retraining, disjoint calibration/policy windows and next-period tests; recent-label recalibration cut cycle-2 Brier loss from **0.00957 to 0.00665**, while a later **3.86% fraud-rate jump** showed that refreshed calibration could still lag sudden base-rate change.

This is the safer lifecycle bullet. Do not turn it into a claim that rolling refresh improved all periods: in cycle 3 refreshed Brier loss worsened to **0.03015** versus **0.02561** for the frozen v1.7 calibrator.

## Strong v1.8 interview story

### 1. Separate prediction, calibration and decision-policy selection

The v1.7 PaySim evidence path is:

- model training: steps **1--445**;
- probability calibration only: **446--519**;
- routing-policy selection only: **520--594**;
- untouched future test: **595--743**.

The validation-stage cutoff is determined from time only, not fraud labels or performance. The predictive feature family is fixed before this experiment.

### 2. Routing remained stable after the stricter split

The policy family is pre-specified as

`priority = P(fraud) × (amount / policy_median_amount)^alpha`, `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`.

Only the later policy-selection stage chooses alpha. Three contiguous policy windows are evaluated at the same **50 reviews per 10,000 transactions**, using worst-window objectives first. Case-first, balanced and value-first objectives all select **alpha=0.25** in the initial lifecycle.

The frozen future result at 50/10k is **61.59% precision / 22.97% fraud-case recall / 77.67% fraud-value recall**.

Good phrasing: “I wanted to know whether the routing decision was stable or just benefiting from reuse of the calibration sample. After separating calibration and routing-selection periods, the same alpha=0.25 compromise was selected and the future operating metrics barely changed.”

### 3. v1.8 asks what happens after deployment time moves forward

The rolling-origin schedule is:

| Cycle | Training | Calibration | Policy selection | Test |
|---:|---|---|---|---|
| 1 | 1--445 | 446--519 | 520--594 | 595--644 |
| 2 | 1--495 | 496--569 | 570--644 | 645--694 |
| 3 | 1--545 | 546--619 | 620--694 | 695--743 |

Each cycle fixes its boundaries from time only. The rolling strategy refits the model on expanding history, fits probability calibration on the declared calibration stage, selects routing alpha only on the preceding policy window and then evaluates the next unseen window.

The comparator keeps the cycle-1 v1.7 model, calibrator and routing policy frozen throughout.

### 4. Recalibration helps cycle 2, then lags a sudden cycle-3 base-rate change

Cycle 2 test prevalence is **0.818%**. The frozen calibrator predicts **2.495%** mean risk, while rolling refresh predicts **1.058%**. Brier loss falls from **0.00957 to 0.00665**.

Cycle 3 is different: fraud prevalence jumps to **3.863%**. The frozen system predicts **3.133%** mean risk, while the recently refreshed system predicts only **1.152%**. Brier loss is **0.02561 frozen versus 0.03015 refreshed**. Ranking PR-AUC changes only modestly, **0.5272 to 0.5331**.

Good phrasing: “I did not assume recalibration would always help. It fixed over-prediction in the second period, then under-predicted badly when prevalence jumped in the third. That told me refresh cadence and base-rate monitoring need their own controls; a recent calibrator can still be stale.”

### 5. Calibration and exact-capacity routing are different contracts

The robust `balanced` policy stays **alpha=0.25 in all three cycles**. At 50 reviews/10k, rolling refresh gives the same precision and case recall as frozen v1.7 in every test cycle. Fraud-value recall is also unchanged except for a small cycle-2 decline from **81.74% to 81.54%**.

This is not a contradiction. Sigmoid calibration is monotone, so large changes in absolute probability do not necessarily change rank. Exact top-k analyst capacity depends primarily on ordering, not whether the probability itself is numerically calibrated.

Good phrasing: “Probability calibration moved a lot, but the balanced investigation queue hardly moved. I therefore monitor score calibration and rank-based queue performance separately.”

### 6. Value-first routing adapts differently, but it is not a new default

The value-first profile selects alpha **0.25 → 0.5 → 1.0** across the three cycles. Cycle 2 slightly worsens out of sample. Cycle 3 is materially different: value-first alpha=1.0 gives **100% precision, 12.77% case recall and 60.95% fraud-value recall**, versus **100% / 12.77% / 40.76%** for frozen alpha=0.25.

The same number of fraud cases enters the queue, but the value-first ranking selects higher-value cases, adding **20.20 percentage points** of fraud-value recall.

Good phrasing: “The late-period value-first policy found more fraud value at the same case count, but the prior cycle did not improve. I treat that as objective-specific out-of-sample evidence, not as justification to replace the balanced default.”

### 7. Exact analyst capacity is the operational contract

Large LightGBM score ties made an earlier narrow scalar threshold unsafe. The current system routes exact top-k capacity and uses only a stable non-label event key to break equal scores.

At frozen alpha=0.25 on steps 595--743:

| Reviews / 10k | Precision | Fraud recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 87.66% | 16.32% | 61.39% |
| 50 | **61.59%** | **22.97%** | **77.67%** |
| 100 | 43.24% | 32.29% | 87.58% |

Do **not** use the old v1.3 **60.1% precision / 25.9% recall / 80.7% value recall** narrow-threshold headline; it is superseded.

### 8. Capacity saturation is not the same as model failure

At 50 reviews/10k, frozen alpha=0.25 value recall is **75.45% → 81.74% → 40.76%** across the three rolling test windows. In the final high-fraud window every admitted case is fraud, yet case recall is only **12.77%**.

A perfect-precision queue can still miss most fraud when fraud arrival rate exceeds analyst capacity.

### 9. Keep negative recipient evidence

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
- Moniepoint fraud prevalence, traffic or analyst capacity;
- measured production latency/SLA;
- a real analyst-confirmation rate;
- confirmed mule-account detection from PaySim;
- simulator balance-field PR-AUC (~0.995) as a realistic fraud headline;
- that PaySim validates real delayed-label refresh timing;
- that rolling refresh improves every period;
- that cycle-3 value-first alpha=1.0 should be the production default.

PaySim may be described only as a **public synthetic external benchmark** used to demonstrate scalable point-in-time feature engineering, temporal evaluation, calibration diagnostics, rolling-origin lifecycle testing and Fraud Ops routing.

PaySim does not contain real label-maturity or investigation-completion timestamps. The rolling-refresh audit is an as-of upper-bound assuming earlier labels are available by the next refresh. The controlled 120k stream remains the source of explicit delayed-label and verification-bias evidence.

## Short interview summary

“I treated fraud detection as a temporal decisioning and operations problem rather than an AUC exercise. On the full 6.36M-row PaySim benchmark, I separated model training, calibration, routing-policy selection and future evaluation, then extended that contract into three rolling-origin cycles. The balanced `P(fraud) × amount^0.25` queue stayed stable, but calibration did not: refresh corrected cycle-2 over-prediction and then lagged a sharp cycle-3 fraud-rate increase. A separately selected value-first policy captured 60.95% versus 40.76% of fraud value in the final capacity-saturated period, but did not improve the preceding cycle, so I kept it as a separate objective rather than replacing the balanced default. The project also stress-tests unseen fraud, delayed labels, verification bias, queue saturation and negative recipient signals.”
