# Moniepoint application notes — current evidence

Use these results only as **portfolio / benchmark evidence**. The controlled 120k stream and PaySim are synthetic; do not present any number as Moniepoint production impact, prevented loss, measured staffing, or real analyst performance.

## Recommended CV bullets

Use at most two bullets on a one-page CV. The strongest default pair is the causal full-PaySim decisioning bullet plus the emerging-fraud stress-test bullet.

### Option A — full PaySim + causal analyst capacity

Built and validated a time-aware fraud decisioning pipeline on the full **6.36M-row PaySim** benchmark using DuckDB point-in-time SQL and GitHub Actions; separated model training, calibration and routing-policy selection in time, then implemented a future-safe seen-so-far analyst backlog reaching **50.9% precision, 19.0% fraud recall and 64.1% fraud-value recall at 50 reviews per 10k transactions**.

This is now the preferred general PaySim routing bullet. The backlog consumes the same 617 reviews as the older whole-window top-k benchmark but never uses later-step scores to decide an earlier review.

If space permits and the role values Fraud Ops, add that the causal backlog had **88.2% queue overlap** with the retrospective batch queue and explicitly measured review delay rather than treating deferred review as free.

### Option B — emerging fraud + Fraud Ops

Stress-tested a fraud pattern absent from all supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; a fixed exploit-explore queue preserved **80.9% fraud-value recall** at 200 reviews/10k while adding **40.2% new-attack recall**.

### Option C — feedback + label maturity

Built an as-of fraud learning loop linking anomaly alerts, analyst review and retraining: **100 anomaly-ranked reviews yielded 95 simulated novel confirmations and raised later novel-fraud recall from 0% to 89.3%**; separately showed that an invalid instant-label backtest produced **88.4%** novel recall where a 7-day mature-label view remained at **0%**.

Option C must retain wording such as `simulated` or `case study` because the confirmation yield is intentionally strong in the controlled attack design.

### Option D — rolling lifecycle + calibration drift

Built a three-cycle rolling-origin fraud model lifecycle on the full **6.36M-row PaySim** benchmark, with expanding-history retraining, disjoint calibration/policy windows and next-period tests; recent-label recalibration cut cycle-2 Brier loss from **0.00957 to 0.00665**, while a later **3.86% fraud-rate jump** showed that refreshed calibration could still lag sudden base-rate change.

### Option E — policy-release uncertainty

Added a paired time-block bootstrap release gate with family-wise error control; a value-first candidate improved retrospective fraud-value recall by **20.2 pp**, but the governed result remained `KEEP_INCUMBENT` because its case-recall guardrail did not pass and the decision was sensitive to the temporal-dependence assumption.

This is a retrospective completed-window policy-comparison result, not a causal online queue result.

## Strong interview story

### 1. Separate model fitting, calibration and policy selection

The initial PaySim lifecycle is:

- model training: steps **1–445**;
- probability calibration: **446–519**;
- routing-policy selection: **520–594**;
- future evaluation: **595–743**.

The split depends only on time. Future labels do not select the feature family, calibrator, or routing alpha.

The routing family is

`priority = P(fraud) × (amount / policy_median_amount)^alpha`, `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`.

The initial robust policy-selection stage chooses `alpha=0.25`.

### 2. Correct the capacity backtest when the operational interpretation changes

An earlier exact-capacity evaluation ranked the entire future window and then selected the top `K`. It did not use fraud labels for ranking, but it could use a later transaction score to determine whether an earlier transaction occupied a review slot.

That result is now labelled **retrospective whole-window batch benchmark**:

**61.59% precision / 22.97% fraud recall / 77.67% fraud-value recall at 50 reviews per 10k.**

Do not use those numbers as an online routing headline.

v1.11 adds two causal comparators with the same 617-review budget:

| Contract | Precision | Fraud recall | Fraud-value recall | Queue overlap vs batch |
|---|---:|---:|---:|---:|
| retrospective batch | 61.59% | 22.97% | 77.67% | 100% |
| seen-so-far backlog | **50.89%** | **18.98%** | **64.10%** | **88.17%** |
| current-step-only | 23.01% | 8.59% | 27.42% | 39.87% |

A strong explanation is:

> “The old fixed-capacity metric was label-safe but had future-score hindsight. I kept it as a retrospective upper benchmark and built two causal queues. The seen-so-far backlog recovered much of the performance without seeing future scores, so I now use that as the safer Fraud Ops benchmark.”

### 3. Treat backlog delay as a cost

At 50 reviews/10k, the backlog has mean review delay **7.72 PaySim steps**, p90 **28**, and maximum **92**. Reviewed fraud cases have mean delay **8.50 steps** and p90 **18.4**.

Do not translate these into a claimed production SLA. PaySim does not model a real analyst service process.

A strong explanation is:

> “I did not let a backlog recover ranking quality for free. I recorded the delay distribution, because a queue that waits longer can improve prioritisation but may be operationally unacceptable.”

### 4. Calibration and ranking are different contracts

In the rolling audit, cycle-2 test prevalence is **0.818%**. The frozen calibrator predicts **2.495%** mean risk, while rolling refresh predicts **1.058%** and lowers Brier loss from **0.00957 to 0.00665**.

Cycle 3 is different: fraud prevalence rises to **3.863%**. The refreshed system predicts only **1.152%**, and Brier loss is **0.03015** versus **0.02561** for the frozen calibrator.

A monotone calibration change can move absolute probability a lot without moving top-k ranking much. Monitor probability quality and queue performance separately.

### 5. Do not promote a candidate from one strong point estimate

In the retrospective cycle-3 value-first comparison, fraud-value recall rises by **20.20 pp** and the family-adjusted lower bound remains **+9.78 pp**. The fraud-case-recall lower bound is **−2.98 pp**, outside the fixed −2 pp non-inferiority margin, so the v1.9 gate returns `KEEP_INCUMBENT`.

v1.10 checks 1/3/5/10-step block lengths. The decision remains `KEEP_INCUMBENT` for 1/3/5-step blocks but becomes `PROMOTE` at 10 steps. Therefore the result is marked `DEPENDENCE_SENSITIVE`; the frozen 5-step decision is not changed after seeing the sensitivity result.

### 6. Keep negative evidence

PaySim has no confirmed mule-account label. Standalone recipient fan-in and recipient-intensity signals have future AUC around **0.47–0.49**, and validation-selected thresholds recover **0% future fraud**. Keep these signals as investigation diagnostics rather than presenting them as mule detection.

## Controlled 120k evidence worth discussing

- Point-in-time behavioural history raises known-fraud PR-AUC **0.597 → 0.640** and fraud-value recall **78.1% → 86.1%** in the controlled stream.
- An unseen future-only attack has **0% supervised recall** while a label-free tail detector reaches **93.3% recall at 0.86% legitimate flag rate**.
- At 200 reviews/10k, a fixed 80/20 exploit-explore split gives **80.9% value recall / 40.2% novel recall**, versus **81.8% / 0%** for model-only routing.
- A 7-day mature-label view has **0% novel recall** where an invalid instant-label oracle reaches **88.4%**, demonstrating label-timing leakage.
- Investigation-driven labels create verification bias: known-fraud PR-AUC is **0.584** with risk-triggered labels versus **0.646** with full historical labels.

## Claims not to make

Do not claim:

- real Moniepoint data;
- real prevented or saved loss;
- a real A/B treatment effect;
- Moniepoint fraud prevalence, traffic or analyst capacity;
- measured production latency or service-level performance;
- a real analyst-confirmation rate;
- confirmed mule-account detection from PaySim;
- simulator balance-field PR-AUC as a realistic fraud headline;
- that PaySim validates real delayed-label refresh timing;
- that rolling refresh improves every period;
- that the cycle-3 value-first candidate should be deployed;
- that the retrospective 61.59% / 22.97% / 77.67% batch result is an online queue result.

PaySim is a public synthetic benchmark used to demonstrate scalable point-in-time feature engineering, temporal evaluation, calibration diagnostics, routing, lifecycle testing and evaluation governance.

## Short interview summary

> “I treated fraud detection as a temporal decisioning and operations problem rather than an AUC exercise. On the full 6.36M-row PaySim benchmark I separated training, calibration, policy selection and future evaluation. I then audited my own fixed-capacity backtest and found that whole-window top-k had future-score hindsight. I kept it as a retrospective benchmark and implemented a causal seen-so-far backlog, which at 50 reviews per 10k reached 50.9% precision, 19.0% fraud recall and 64.1% fraud-value recall with an explicit delay distribution. The project also tests calibration drift, policy-release uncertainty, unseen fraud, delayed labels, verification bias and negative recipient signals.”
