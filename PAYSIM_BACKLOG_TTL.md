# PaySim backlog TTL and rescore-workload frontier

## Why this audit exists

v1.13 showed that a causal analyst backlog cannot be treated as a stateless model output. At the two rolling model-release boundaries, an unbounded rescore-on-refresh policy had 42,204 and 108,643 pending cases eligible for rescoring. That policy produced the highest balanced H-mean among the v1.13 handoff variants, but the release workload itself was not controlled.

v1.14 asks a narrower operations question: **can an age TTL materially reduce backlog/rescore workload without materially degrading the governed detection outcomes?**

This is not another model-selection experiment. The model lifecycle, balanced routing objective, rescore-on-refresh handoff, exact continuous analyst capacity and tie-breaking rule are frozen. Only the maximum age of an unreviewed pending case changes.

## Predeclared contract

The TTL grid was fixed before the full-data results were read:

`TTL ∈ {0, 5, 10, 20, 40, infinite}` PaySim steps.

A pending transaction remains eligible at step `t` only while:

`current_step - arrival_step <= TTL`.

Expiry occurs before a model refresh at that step. Therefore a finite TTL can reduce the number of pending cases that must be rescored by the newly released model. Labels are never used to decide expiry or queue membership.

Every TTL uses:

- the same full 6,362,620-row canonical PaySim source;
- the same three rolling model/calibration/policy regimes;
- the same `balanced` routing profile;
- one continuous steps 595--743 evaluation horizon;
- exactly 50 reviews per 10,000 arrivals, giving the same final **617 reviews**;
- rescore-on-refresh for all pending cases that survive the TTL;
- stable non-label `event_key` tie-breaking.

`TTL=infinite` is a locked reproducibility anchor. In PR context it reproduced the v1.13 `rescore_pending` metrics exactly.

## Predeclared operational screen

The screen was fixed before reading the full-data frontier. Relative to `TTL=infinite`, each finite TTL must satisfy all of:

- precision decline no worse than **2 percentage points**;
- fraud-recall decline no worse than **2 percentage points**;
- fraud-value-recall decline no worse than **2 percentage points**;
- total refresh-time rescore workload reduced by at least **50%**.

A row passing both the detection guardrails and workload target is labelled `operational_candidate`.

This is a deterministic engineering screen. It is **not** a statistical promotion test and not a production recommendation.

## Full-data result

All rows use the same 617-review continuous capacity.

| TTL | Precision | Fraud recall | Fraud-value recall | Balanced H-mean | Mean delay | Total refresh rescores | Rescore reduction | Queue overlap vs ∞ | Screen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 23.34% | 8.71% | 27.33% | 0.1321 | 0.00 | 0 | 100.0% | 48.46% | fail detection |
| 5 | 40.52% | 15.11% | 47.39% | 0.2292 | 1.61 | 24,121 | 84.0% | 78.12% | fail detection |
| 10 | 44.73% | 16.69% | 54.86% | 0.2559 | 2.68 | 48,163 | 68.1% | 84.28% | fail detection |
| 20 | 49.92% | 18.62% | 61.57% | 0.2860 | 3.84 | 59,845 | **60.3%** | 88.17% | fail precision |
| 40 | 50.08% | 18.68% | 62.21% | 0.2873 | 6.23 | 87,513 | 42.0% | **95.79%** | fail precision + workload |
| ∞ | **52.67%** | **19.65%** | **63.39%** | **0.3000** | 8.35 | 150,847 | 0% | 100% | reference |

There is **no finite operational candidate** under the predeclared screen.

## The boundary is informative rather than a failed experiment

TTL 20 is the closest row to the intended workload/detection compromise:

- rescore workload falls by **60.33%**, from 150,847 to 59,845 total refresh rescoring operations;
- mean review delay falls from **8.35 to 3.84** steps;
- fraud recall declines only **1.03 pp**;
- fraud-value recall declines only **1.82 pp**;
- but precision declines **2.76 pp**, exceeding the frozen 2 pp guardrail.

The correct result is therefore **do not pass TTL 20 under this rule**. The precision margin is not relaxed after seeing the result.

TTL 40 illustrates the other side of the frontier:

- it preserves **95.79%** of the infinite-TTL review queue;
- fraud recall declines **0.97 pp** and value recall **1.18 pp**;
- precision still declines **2.59 pp**;
- rescore workload falls only **41.99%**, below the frozen 50% target.

So no tested TTL simultaneously meets the declared detection and workload requirements.

## Refresh workload

With no TTL, the eligible pending pool is rescored in full:

- step 645: **42,204** pending cases;
- step 695: **108,643** pending cases;
- total: **150,847** rescoring operations.

TTL 20 reduces the corresponding refresh rescoring counts to:

- step 645: **8,484**;
- step 695: **51,361**;
- total: **59,845**.

TTL 40 gives 21,417 and 66,096, or 87,513 total.

The reduction comes from continuous age expiry before the model refresh, not from using labels or future scores.

## Queue displacement matters

The finite-TTL queue should not be interpreted only through aggregate recall. Relative to infinite TTL, the number of the 617 investigation assignments replaced is:

| TTL | Queue overlap | Replaced assignments |
|---:|---:|---:|
| 0 | 48.46% | 318 |
| 5 | 78.12% | 135 |
| 10 | 84.28% | 97 |
| 20 | 88.17% | 73 |
| 40 | 95.79% | 26 |

This shows why TTL 40 can look close in aggregate metrics while still failing the predeclared operational target: it changes only 26 reviews, but it also leaves a large refresh workload.

## How to interpret `expired_cases`

The output also reports the number and labelled fraud/value composition of cases that eventually age out under each finite TTL. These are retrospective diagnostics only.

**Do not interpret total expired cases as incremental missed investigations caused by TTL.** Under a 50/10k capacity constraint, almost all arrivals are never reviewed even with infinite TTL. A finite TTL simply gives many of those never-selected cases a terminal expired state. The direct comparison of the 617 realised investigation assignments is `ttl_queue_overlap.csv` and is the safer measure of queue displacement.

## What this changes in the project story

The project now distinguishes four separate operational questions:

1. can the model rank future fraud under point-in-time features?;
2. can analyst capacity be evaluated without future-score hindsight?;
3. what happens to a causal pending backlog when a new model is released?;
4. can backlog age be bounded without violating predeclared detection and workload requirements?

For the tested frontier, question 4 has a negative answer: **none of the finite TTLs meets both requirements**.

That negative result is retained rather than tuning the guardrails around the observed frontier.

## Claim boundaries

- PaySim is synthetic mobile-money data, not Moniepoint production data.
- PaySim steps are dataset time units, not a measured analyst SLA.
- Transaction amount is not prevented loss or realised financial benefit.
- Fraud labels are used only for retrospective evaluation, never for TTL expiry or queue selection.
- The rolling lifecycle still uses the v1.8 as-of assumption that prior-period labels are available at the next refresh; PaySim does not contain real investigation-completion or label-maturity timestamps.
- The 2 pp detection margins and 50% workload target are engineering guardrails, not confidence bounds.
- `operational_candidate=False` means the deterministic screen did not pass; it does not prove that every possible TTL or production implementation is inferior.
