# PaySim causal analyst-capacity audit — v1.11

## Why this audit exists

The earlier PaySim routing evidence uses an exact-capacity whole-window top-k rule. That rule never uses fraud labels to rank transactions, but it sees the scores of every transaction in the evaluation window before deciding which rows occupy the fixed analyst budget.

That is valid as a **retrospective batch benchmark**. It is not a causal online queue simulation because an early transaction can lose a review slot to a higher-scoring transaction that arrives later in the same evaluation window.

v1.11 makes this boundary explicit and adds two causal comparators that never use later-step scores.

## Frozen model and policy

The predictive system is unchanged from the v1.7 stage-separated reference:

1. model training: steps 1–445;
2. probability calibration: steps 446–519;
3. routing-policy selection: steps 520–594;
4. future audit: steps 595–743.

The feature family remains `transaction_plus_relational`. The robust policy-selection stage still chooses `alpha=0.25` for `case_first`, `balanced`, and `value_first`, so the three profile rows are identical in this audit.

The only new variable is how a fixed analyst-capacity budget is allocated through future time.

## Three routing contracts

### Retrospective whole-window batch

The full future window is ranked once and the top `K` rows are selected, where

`K = floor(alerts_per_10k × N / 10,000)`.

This is useful as a hindsight upper benchmark for a fixed score and fixed total review count. It should not be described as an online operating result.

### Causal current-step-only

PaySim steps are processed in ascending order. At every step the cumulative review entitlement is

`floor(alerts_per_10k × cumulative_transactions / 10,000)`.

Newly earned slots are spent immediately on the highest-scoring transactions in the current step. A later step can never change an earlier selection. Fractional capacity is retained through the cumulative floor.

This is a strict low-latency comparator. It does not permit an unreviewed transaction from an earlier step to remain in a backlog.

### Causal seen-so-far backlog

The same cumulative entitlement is used, but all observed, not-yet-reviewed transactions remain in a priority backlog. When a new review slot becomes available, it is assigned to the highest-scoring transaction among all rows observed so far.

This permits deferred review but still cannot use any later-step score. The audit records review delay as an explicit cost of this flexibility.

All three contracts consume exactly the same final number of reviews. Equal scores use the same stable non-label `event_key` tie-breaker.

## Full 6.36M-row result

GitHub Actions runs the audit on the canonical **6,362,620-row PaySim** data with **8,213 fraud cases**. The future audit contains **123,580 transactions across steps 595–743**.

At the reference capacity of **50 reviews per 10,000 transactions**, every contract uses exactly **617 reviews**.

| Routing contract | Precision | Fraud-case recall | Fraud-value recall | Queue overlap vs batch | Replacement rate |
|---|---:|---:|---:|---:|---:|
| Retrospective whole-window batch | **61.59%** | **22.97%** | **77.67%** | 100% | 0% |
| Causal seen-so-far backlog | **50.89%** | **18.98%** | **64.10%** | **88.17%** | **11.83%** |
| Causal current-step-only | **23.01%** | **8.59%** | **27.42%** | **39.87%** | **60.13%** |

The whole-window batch result is therefore materially optimistic if it is interpreted as an online queue. The seen-so-far backlog recovers much of the gap without future-score access, but it does not fully recover the retrospective result.

Relative to the batch queue, the backlog causal contract changes 11.83% of the 617 review assignments. Those queue substitutions reduce captured fraud cases by 66 and reduce captured **synthetic transaction amount associated with fraud**. The amount difference is intentionally not interpreted as money saved or lost.

## Backlog delay

The causal backlog result is not free. At 50 reviews/10k:

- mean review delay: **7.72 PaySim steps**;
- p50 review delay: **0 steps**;
- p90 review delay: **28 steps**;
- maximum review delay: **92 steps**;
- mean delay among reviewed fraud cases: **8.50 steps**;
- p90 delay among reviewed fraud cases: **18.4 steps**.

A transaction with zero delay was reviewed in its arrival step. Positive delay means it remained in the seen-so-far backlog until later cumulative capacity became available.

PaySim steps are kept as dataset steps in the evidence. The project does not convert these values into a claimed real investigation SLA.

## Capacity frontier

For the `balanced` profile, which is identical to the other two profiles here because all select `alpha=0.25`:

| Reviews / 10k | Contract | Precision | Fraud recall | Fraud-value recall | Queue overlap vs batch |
|---:|---|---:|---:|---:|---:|
| 10 | batch | 100.00% | 7.44% | 33.93% | 100% |
| 10 | backlog | 86.18% | 6.41% | 27.23% | 75.61% |
| 10 | current-step | 51.22% | 3.81% | 13.13% | 26.83% |
| 25 | batch | 87.66% | 16.32% | 61.39% | 100% |
| 25 | backlog | 69.16% | 12.88% | 47.87% | 79.22% |
| 25 | current-step | 34.42% | 6.41% | 21.68% | 33.44% |
| 50 | batch | 61.59% | 22.97% | 77.67% | 100% |
| 50 | backlog | 50.89% | 18.98% | 64.10% | 88.17% |
| 50 | current-step | 23.01% | 8.59% | 27.42% | 39.87% |
| 100 | batch | 43.24% | 32.29% | 87.58% | 100% |
| 100 | backlog | 36.28% | 27.09% | 73.34% | 92.39% |
| 100 | current-step | 15.06% | 11.25% | 30.92% | 56.36% |

The gap narrows as analyst capacity increases. At 100 reviews/10k, the backlog queue overlaps 92.39% of the retrospective batch queue; at 10 reviews/10k it overlaps 75.61%.

## What v1.11 changes in interpretation

The old **61.59% precision / 22.97% fraud recall / 77.67% fraud-value recall** result remains reproducible and useful, but its label is now **retrospective whole-window batch benchmark**.

For an application or interview that asks how the queue could operate without future information, the safer 50/10k benchmark is the **seen-so-far backlog** result: **50.89% precision / 18.98% fraud recall / 64.10% fraud-value recall**, together with its review-delay distribution.

The strict current-step result is retained as a low-latency stress comparator rather than used as the sole online estimate.

This correction does not imply fraud-label leakage in the earlier benchmark. The issue is cross-step score hindsight in capacity allocation.

## Reproducibility controls

The dedicated GitHub Actions workflow:

- downloads the canonical PaySim parquet shards;
- verifies 6,362,620 rows, 8,213 fraud cases, and steps 1–743;
- rebuilds strict prior-step features;
- refits the frozen stage-separated reference model and calibrator;
- reselects routing alpha only from the policy-selection stage;
- evaluates future steps 595–743 under 10/25/50/100 reviews per 10k;
- requires batch and both causal contracts to consume identical final capacity;
- verifies in tests that backlog reviews never occur before transaction arrival;
- writes aggregate comparison and schedule artifacts without committing raw PaySim rows.

## Evidence boundary

PaySim is synthetic mobile-money data. It has no real analyst roster, review service times, chargeback maturity, intervention effect, or investigation-completion process. The backlog delay is therefore a dataset-step audit, not a real service-level measurement.

A production queue would need an explicit service model for analyst concurrency, ageing/expiry, priority refresh, review duration, escalation, and label maturity. v1.11 addresses future-score hindsight first; it does not claim to simulate all of those operational processes.
