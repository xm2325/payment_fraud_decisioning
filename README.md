# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project centred on **temporal modelling, point-in-time SQL/Python features, calibrated risk scores, analyst-capacity routing, emerging-fraud detection, delayed labels, model refresh and release governance**.

The repository has two evidence layers:

- a transparent **120,000-transaction synthetic payment stream** for controlled stress tests such as unseen attacks, analyst feedback, verification bias and label delay;
- a reproducible **6,362,620-row PaySim benchmark** for large-scale point-in-time SQL, temporal validation, calibration, routing and lifecycle audits.

Neither source is Moniepoint production data. No result below is a production impact, prevented-loss or staffing claim.

## Current result snapshot — v1.11

### Full 6.36M-row PaySim benchmark

GitHub Actions verifies **6,362,620 transactions, 8,213 fraud cases and steps 1–743**, builds strict prior-step DuckDB features and evaluates a balance-free `transaction_plus_relational` LightGBM reference.

The model lifecycle is time ordered:

1. model training: steps **1–445**;
2. probability calibration: **446–519**;
3. routing-policy selection: **520–594**;
4. future evaluation: **595–743**.

The initial robust routing profile uses

`priority = P(fraud) × (amount / policy_median_amount)^alpha`

with `alpha=0.25`, selected only from the policy-selection stage.

### v1.11 causal analyst-capacity correction

Earlier versions evaluated exact analyst capacity by ranking an entire future evaluation window and selecting the top `K`. That rule is label-free, but it has **future-score hindsight**: an early transaction can lose a review slot to a higher score that arrives later in the evaluation window.

v1.11 therefore keeps the old rule only as a **retrospective whole-window batch benchmark** and adds two causal routing contracts. Both process PaySim steps in ascending order and never compare an observed transaction with a later-step score. Both consume exactly the same final review budget as the batch benchmark.

At **50 reviews per 10,000 transactions**, all three contracts use **617 reviews**:

| Routing contract | Precision | Fraud-case recall | Fraud-value recall | Queue overlap vs batch |
|---|---:|---:|---:|---:|
| Retrospective whole-window batch | **61.59%** | **22.97%** | **77.67%** | 100% |
| Causal seen-so-far backlog | **50.89%** | **18.98%** | **64.10%** | **88.17%** |
| Causal current-step-only | **23.01%** | **8.59%** | **27.42%** | **39.87%** |

The backlog contract lets observed, unreviewed transactions remain eligible when later analyst capacity becomes available, but it still cannot use future scores. It recovers much of the strict current-step loss without reaching the retrospective batch result.

Backlog flexibility has a measurable delay cost at 50 reviews/10k: mean review delay is **7.72 PaySim steps**, p90 is **28 steps**, and the maximum is **92 steps**. These are dataset-step diagnostics, not a real investigation service-level agreement.

The old **61.59% / 22.97% / 77.67%** precision/case-recall/value-recall result remains reproducible, but it should no longer be presented as an online routing headline. For a future-safe Fraud Ops example, use the **50.89% / 18.98% / 64.10%** backlog result together with its delay boundary.

See `PAYSIM_ONLINE_CAPACITY.md` for the full audit.

## v1.9–v1.10 policy release governance

The rolling lifecycle also compares a frozen v1.7 routing policy with refreshed candidates. v1.9 adds a paired circular time-block bootstrap and a pre-declared promotion gate across six cycle/profile comparisons.

The strongest candidate is cycle-3 `value_first`: its retrospective completed-window fraud-value recall increases by **20.20 percentage points**, and the Bonferroni-adjusted one-sided lower bound remains **+9.78 pp**. However, the fraud-case-recall lower bound is **−2.98 pp**, outside the fixed **−2 pp** non-inferiority margin, so the formal decision is **`KEEP_INCUMBENT`**.

v1.10 then checks dependence assumptions using **1/3/5/10-step** circular blocks. The v1.9 5-step result must be reproduced numerically before the sensitivity audit is accepted.

Five of six comparisons are `ROBUST_KEEP_INCUMBENT`. Cycle-3 `value_first` is `DEPENDENCE_SENSITIVE`: it remains `KEEP_INCUMBENT` for 1/3/5-step blocks but changes to `PROMOTE` for 10-step blocks. The frozen v1.9 decision is not changed after seeing this sensitivity result.

These v1.9/v1.10 analyses are **retrospective completed-window policy audits**, not literal simulations of which transaction would have entered an online analyst queue. v1.11 handles the causal queue-execution question separately.

See `PAYSIM_POLICY_PROMOTION.md` and `PAYSIM_PROMOTION_SENSITIVITY.md`.

## v1.8 rolling model lifecycle

v1.8 moves the evaluation origin forward while preserving disjoint training, calibration and policy-selection stages:

| Cycle | Model training | Calibration | Policy selection | Test |
|---:|---|---|---|---|
| 1 | 1–445 | 446–519 | 520–594 | 595–644 |
| 2 | 1–495 | 496–569 | 570–644 | 645–694 |
| 3 | 1–545 | 546–619 | 620–694 | 695–743 |

Cycle 2 shows why recalibration can help: actual fraud prevalence is **0.818%**, the frozen v1.7 calibrator predicts **2.495%** mean risk, while rolling refresh predicts **1.058%** and reduces Brier loss from **0.00957 to 0.00665**.

Cycle 3 shows the opposite failure mode. Fraud prevalence rises to **3.863%**; the refreshed calibrator predicts **1.152%**, and Brier loss is **0.03015** versus **0.02561** for the frozen calibrator. Recent calibration can still lag a sharp base-rate shift.

PaySim does not contain real fraud-label maturity or investigation-completion timestamps. Later cycles may use labels from completed earlier periods, so this is an as-of benchmark under labels being available by the next refresh, not production delayed-label evidence.

See `PAYSIM_ROLLING_REFRESH.md`.

## Controlled 120k stress tests

The smaller controlled stream is retained because it can test failure modes that PaySim does not represent explicitly:

- point-in-time velocity/history increases known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at the declared operating point;
- the supervised classifier gets **0% recall** on a future-only shared-device attack, while a label-free tail detector reaches **93.3% recall at 0.86% legitimate flag rate**;
- at **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**, while a fixed 80/20 exploit-explore queue gets **80.9% / 40.2%**;
- a 7-day mature-label view remains at **0% novel recall**, while an invalid instant-label oracle reaches **88.4%**, showing the effect of label-timing leakage;
- risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels, exposing verification bias.

These are controlled simulator stress tests, not production estimates.

## Why exact analyst capacity is explicit

Large LightGBM score ties make a narrow scalar threshold a poor contract for a fixed analyst budget. The project therefore uses explicit reviews-per-10,000 capacity and a stable non-label `event_key` for ties.

v1.11 adds a second requirement: **capacity allocation through time must also be causal** if the result is described as an online queue.

This creates three different objects that should not be mixed:

- probability quality — calibration and ranking diagnostics;
- retrospective fixed-budget ranking — whole-window top-k benchmark;
- causal queue execution — current-step or seen-so-far backlog routing.

The previously documented v1.3 **60.1% precision / 25.9% recall / 80.7% value recall** narrow-threshold headline is superseded and should not be reused.

## Recipient / mule audit: retained negative evidence

PaySim has no confirmed mule-account label. Standalone prior-step recipient signals remain investigation diagnostics rather than mule classifiers. Recipient fan-in AUC is about **0.493**, the composite recipient-intensity score about **0.467**, and validation-selected recipient thresholds recover **0% future fraud**. The negative result remains visible.

## Architecture

```text
transaction
   -> strict prior-step transaction + relational history
   -> balance-free supervised model
   -> calibration-only temporal stage
   -> policy-selection-only temporal stage
   -> calibrated fraud probability + amount-aware priority
   -> causal cumulative analyst capacity
        -> current-step micro-batch comparator
        -> seen-so-far priority backlog
   -> completed-period monitoring / policy comparison
   -> rolling model + calibration + policy refresh
```

The controlled 120k stream retains a separate anomaly/exploration lane for future-only attacks and remains the source of explicit delayed-label and verification-bias evidence.

## Reproducibility controls

- Equal-timestamp simulator events cannot use one another as history.
- SQL/Python point-in-time parity is executed in tests.
- Full PaySim features use strict prior-step DuckDB windows.
- Stable non-label event keys make loading and exact-capacity tie-breaking deterministic.
- Model training, calibration, routing-policy selection and future evaluation have explicit temporal boundaries.
- Rolling refresh boundaries depend only on time and are checked for strict ordering and expanding history.
- v1.9 uses a frozen family-wise policy-promotion rule; v1.10 checks dependence assumptions without changing that rule.
- v1.11 tests that causal queue decisions never occur before transaction arrival and that every comparator consumes the same final total review capacity.
- Full-data workflows download canonical PaySim in GitHub Actions and commit only aggregate results; raw PaySim rows are not committed.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
uvicorn fraud_decisioning.api:app --app-dir src --reload
```

Full PaySim workflows are defined under `.github/workflows/` and download the canonical parquet shards during CI.

## Key evidence

- `RESULTS.md` — controlled and PaySim result narrative.
- `PAYSIM_BENCHMARK.md` — canonical 6.36M-row benchmark contract.
- `PAYSIM_MONITORING.md` — monitoring and fixed-capacity diagnostics.
- `PAYSIM_ROUTING_PROFILES.md` — probability/value routing trade-off.
- `PAYSIM_ROUTING_ROBUSTNESS.md` — validation-window routing robustness.
- `PAYSIM_STAGE_SEPARATION.md` — disjoint calibration and routing-policy stages.
- `PAYSIM_ROLLING_REFRESH.md` — frozen-versus-refresh rolling-origin audit.
- `PAYSIM_POLICY_PROMOTION.md` — uncertainty-aware retrospective policy gate.
- `PAYSIM_PROMOTION_SENSITIVITY.md` — block-length dependence sensitivity.
- `PAYSIM_ONLINE_CAPACITY.md` — causal analyst-capacity and backlog audit.
- `APPLICATION_NOTES.md` and `APPLICATION_NOTES_V1_9.md`–`APPLICATION_NOTES_V1_11.md` — claim-safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source and claim boundaries.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

## Data honesty

The 120k simulator deliberately contains a future-only attack that is detectable by historical anomaly signals, so its discovery results are controlled method stress tests. PaySim is also synthetic mobile-money data.

Real deployment would require real fraud-label maturity, analyst service times, queue ageing/expiry, intervention outcomes, customer impact, device/network quality, operating prevalence and staffing constraints. Synthetic transaction amount is not interpreted as prevented loss.
