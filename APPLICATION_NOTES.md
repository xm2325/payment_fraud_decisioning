# Moniepoint fraud Data Scientist application notes — v1.6

Use these as evidence from a **portfolio case study**, not as claims of direct Moniepoint or production fraud experience.

## Best CV bullets

Use at most two on a one-page CV.

### Option A — full-data fraud decisioning + SQL

Built a reproducible fraud-decisioning benchmark on the full **6.36M-row PaySim** mobile-money simulator using DuckDB point-in-time SQL, calibrated LightGBM and GitHub Actions; replaced a tie-sensitive threshold with exact review-capacity routing and, at **50 reviews/10k**, the balance-free relational model achieved **64.3% precision / 24.0% fraud recall / 71.7% fraud-value recall**, versus **42.5% / 15.8% / 70.4%** for a simple amount/type rule.

### Option B — value-aware Fraud Ops routing

Designed a validation-selected review priority `P(fraud) × amount^alpha` under a fixed analyst budget; a pre-specified alpha grid selected **alpha=0.25** before future evaluation, and at **50 reviews/10k** the frozen PaySim policy increased fraud-value recall from **71.7% to 77.7%** versus pure risk ranking while retaining **61.8% precision and 23.0% fraud-case recall**.

### Option C — surge-capacity governance

Stress-tested a score-load capacity guardrail on the full **6.36M-row PaySim** benchmark: the frozen model showed a **2.41× validation score-tail load** in the final future window; under a stated **50→100 reviews/10k** flex-capacity scenario, fraud-case recall increased **12.8%→25.5%** and fraud-value recall **40.8%→81.4%** while precision remained **99.3%**.

For Option C, keep wording such as **PaySim benchmark**, **stress-tested** and **scenario**. Do not turn the assumed 50→100 capacity into a staffing recommendation.

### Option D — emerging fraud + feedback loop

Stress-tested a fraud pattern deliberately absent from supervised training/validation data: the classifier had **0% unseen-attack recall**, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**; anomaly-ranked analyst feedback then supplied simulated confirmations for later supervised retraining.

## Strong interview stories

### 1. A high AUC can be a data problem

PaySim balance derivatives lift PR-AUC to around **0.995**. Instead of using that as a headline, the project treats it as evidence that simulator accounting mechanics are unusually informative and reports a balance-free benchmark. This is a useful answer to “How do you check leakage or unrealistic features?”

### 2. An operating threshold is not the same as an operations budget

Tree scores have ties. A threshold chosen for a nominal narrow false-positive rate can flag a whole tied block or under-use capacity if the threshold is moved above it. The project therefore separates:

- scalar thresholds for diagnostic/governance checks;
- exact top-k routing for Fraud Ops capacity.

Equal scores are broken only with a stable non-label event key.

### 3. Case recall and fraud-value recall are different business objectives

At 50 reviews/10k on future PaySim:

- pure `P(fraud)` gives **64.3% precision / 24.0% case recall / 71.7% value recall**;
- `P(fraud) × amount` gives **56.4% / 21.0% / 77.0%**;
- validation-selected `alpha=0.25` gives **61.8% / 23.0% / 77.7%**.

The selected compromise is preferable to saying one ranking score is universally best. Fraud value is highly concentrated in PaySim, so value recall is always discussed beside case recall and an amount-based baseline.

### 4. Model quality and review capacity are different failure modes

In the final PaySim window the fixed 50/10k queue has **100% precision** but only **12.8% fraud-case recall**. The queue is saturated with true fraud; improving classification precision cannot create more review slots. This motivates a separate capacity-control layer.

### 5. A monitoring trigger still needs governance sensitivity

The pre-specified 1.5× score-tail trigger fires in all three future windows, but the first is only **1.502×**. A sensitivity grid shows that 2.0× would fire only in the final window and 2.5× would never fire. Do not pick 2.0× retrospectively and call it optimal; explain that a production trigger needs business-owned capacity costs, mature labels and prospective validation.

### 6. Negative fraud features should stay negative

Standalone PaySim recipient/fan-in signals are near random and get 0% future fraud recall in the audited setup. Because PaySim has no confirmed mule label, the project keeps these as investigation signals and does not invent a mule-detection claim.

### 7. Label collection changes what the model learns

In the internal synthetic verification-bias stress test, risk-triggered follow-up over-represents fraud and under-covers some typologies. A random audit lane improves coverage. The lesson is to treat investigation policy as part of the data-generating process, not simply assume observed labels are representative.

### 8. Delayed labels make temporal evaluation non-negotiable

A 7-day mature-label view misses the controlled new attack while an invalid instant-label oracle appears dramatically better. Use this to explain why fraud model retraining, backtesting and performance monitoring must respect chargeback/investigation maturity.

## Take-home / technical interview preparation

Use `TAKE_HOME_WALKTHROUGH.md` and the SQL files to practise:

- point-in-time rolling aggregates;
- equal-timestamp / same-step leakage prevention;
- recipient/fan-in investigations;
- rule-vs-model comparisons at the same review budget;
- fixed-threshold versus exact-capacity semantics;
- temporal splits and label maturity;
- typology sizing by transaction count and value;
- explaining why a negative feature audit is still a useful result.

## Claims not to make

Do not claim:

- real Moniepoint data or direct Moniepoint impact;
- real prevented loss or saved money;
- actual Moniepoint fraud prevalence, traffic or analyst staffing;
- a production SLA measured from this project;
- confirmed PaySim mule accounts;
- that the 1.5×, 2.0× or 2.5× score-load trigger is production-optimal;
- that extra review capacity can be created instantly;
- that PaySim precision/recall will transfer to another market.

Safe framing is: **public synthetic external benchmark + controlled synthetic stress tests + reproducible decision-policy engineering**.
