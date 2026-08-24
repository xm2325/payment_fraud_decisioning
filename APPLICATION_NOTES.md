# Moniepoint application notes — v1.7

Use these results only as **portfolio / benchmark evidence**. The 120k stream and PaySim are synthetic; do not present any number as Moniepoint production impact, prevented loss or real analyst performance.

## Recommended CV bullets

Use at most **two** bullets on a one-page CV. The strongest pairing is the stage-separated full-PaySim bullet plus the emerging-fraud bullet.

### Option A — full PaySim + future-safe decision policy

Built and validated a time-aware fraud decisioning pipeline on the full **6.36M-row PaySim** benchmark using DuckDB point-in-time SQL and GitHub Actions; separated model training, probability calibration, routing-policy selection and untouched future evaluation into ordered temporal stages, with a validation-selected `P(fraud) × amount^0.25` review policy reaching **61.6% precision, 23.0% fraud-case recall and 77.7% fraud-value recall at 50 reviews per 10k transactions**.

This is the preferred PaySim bullet because the routing policy is selected without reusing the calibration-stage labels and all reported future metrics use an exact analyst-capacity contract.

### Option B — emerging fraud + Fraud Ops

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

### Option C — feedback + label maturity

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the controlled attack design.

## Strong v1.7 interview story

### 1. Separate prediction, calibration and decision-policy selection

The final PaySim evidence path is:

- model training: steps **1--445**;
- probability calibration only: **446--519**;
- routing-policy selection only: **520--594**;
- untouched future test: **595--743**.

The validation-stage cutoff is determined from time only, not fraud labels or performance. The predictive feature family is fixed before this experiment.

### 2. Routing remained stable after the stricter split

The policy family is pre-specified as

`priority = P(fraud) × (amount / policy_median_amount)^alpha`, `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`.

Only the later policy-selection stage chooses alpha. Three contiguous policy windows are evaluated at the same **50 reviews per 10,000 transactions**, using worst-window objectives first. Case-first, balanced and value-first objectives all still choose **alpha=0.25**.

The selected future result at 50/10k is **61.59% precision / 22.97% fraud-case recall / 77.67% fraud-value recall**. This is almost unchanged from v1.6 despite removing calibration-stage labels from policy selection.

Good phrasing: “I wanted to know whether the routing decision was stable or just benefiting from reuse of the calibration sample. After separating calibration and routing-selection periods, the same alpha=0.25 compromise was selected and the untouched future operating metrics barely changed.”

### 3. Probability calibration did not remain stable

Calibration-stage fraud prevalence is **1.120%** and mean predicted risk is **1.125%**. But with that calibrator frozen:

- policy-selection prevalence is **0.494%**, while mean predicted risk is **1.662%**;
- future prevalence is **1.338%**, while mean predicted risk is **2.533%**.

This is an important negative result. Stage separation makes the evaluation cleaner; it does not make one-time calibration portable through time.

Good phrasing: “The ranking policy was robust, but the absolute probabilities were not. I would therefore monitor calibration and analyst-capacity routing as separate operational contracts, with recalibration requiring mature labels and an as-of governance process.”

### 4. Exact analyst capacity is the operational contract

Large LightGBM score ties made an earlier narrow scalar threshold unsafe. The current system routes exact top-k capacity and uses only a stable non-label event key to break equal scores.

At frozen alpha=0.25 on future PaySim:

| Reviews / 10k | Precision | Fraud recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 87.66% | 16.32% | 61.39% |
| 50 | **61.59%** | **22.97%** | **77.67%** |
| 100 | 43.24% | 32.29% | 87.58% |

Do **not** use the old v1.3 **60.1% precision / 25.9% recall / 80.7% value recall** narrow-threshold headline; it is superseded.

### 5. Capacity saturation is not the same as model failure

At 50 reviews/10k, alpha=0.25 future value recall is **75.45% → 81.74% → 40.76%** across three windows. In the final high-fraud window every admitted case is fraud, yet case recall is only **12.77%**.

That is a strong operational example: a perfect-precision queue can still miss most fraud when fraud arrival rate exceeds analyst capacity.

### 6. Keep negative recipient evidence

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
- simulator balance-field PR-AUC (~0.995) as a realistic fraud headline.

PaySim may be described only as a **public synthetic external benchmark** used to demonstrate scalable point-in-time feature engineering, temporal evaluation, calibration diagnostics and Fraud Ops routing.

## Short interview summary

“I treated fraud detection as a temporal decisioning and operations problem rather than an AUC exercise. On the full 6.36M-row PaySim benchmark, I separated model training, calibration, routing-policy selection and future evaluation. The routing compromise `P(fraud) × amount^0.25` survived the stricter validation-only selection and at 50 reviews per 10k reached 61.6% precision, 23.0% fraud-case recall and 77.7% fraud-value recall. The same experiment showed that absolute calibration drifted badly between periods, so I would govern probability calibration separately from rank-based analyst routing. The project also stress-tests unseen fraud, delayed labels, verification bias, queue saturation and negative recipient signals.”
