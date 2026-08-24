# PaySim cross-refresh backlog handoff audit — v1.13

## Why this audit exists

v1.11 replaced whole-window hindsight routing with a causal seen-so-far backlog. v1.12 then used that causal queue inside each rolling policy-promotion test. One boundary remained: every rolling test window started with an empty evaluation backlog.

A deployed fraud queue does not normally become empty just because a new model is released. Pending cases still exist, and a release process must decide what happens to them.

v1.13 keeps one continuous analyst-capacity entitlement across PaySim steps **595–743** and tests backlog hand-off at the rolling refresh boundaries **645** and **695**.

## Fixed evaluation contract

- Data: canonical PaySim, **6,362,620 transactions / 8,213 fraud cases / steps 1–743**.
- Future operations horizon: steps **595–743**, **123,580 transactions**.
- Routing profile: the default `balanced` policy only.
- Capacity: **50 reviews per 10,000 transactions**.
- Continuous final entitlement: **617 reviews** for every strategy.
- Refresh boundaries: steps **645** and **695**.
- All arrivals are processed forward in time.
- A new model score cannot be used before that model's release boundary.

The balanced routing alpha is **0.25 in all three regimes**; the model, calibrator and policy amount scale still change at refresh.

## Strategies

### `frozen_incumbent`

Keep the cycle-1 model/calibrator/policy for the full horizon. This is the v1.11 continuous causal backlog reference.

### `retain_old_scores`

Switch to the newly released rolling system for new arrivals, but pending cases keep whichever score they already had.

### `rescore_pending`

At each refresh boundary, recompute the routing score of every pending case with the newly released system. New arrivals are also scored by the current system.

### `drop_pending`

Expire all unresolved cases at each model refresh. Capacity itself is not reset; later arrivals can still consume the same final continuous entitlement.

## Full-data result

All four strategies use exactly **617 reviews**.

| Strategy | Precision | Fraud recall | Fraud-value recall | Balanced H-mean | Mean review delay | p90 delay |
|---|---:|---:|---:|---:|---:|---:|
| frozen incumbent | 50.89% | 18.98% | **64.10%** | 0.2929 | 7.72 | 28.0 |
| retain old scores | **53.16%** | **19.83%** | 60.10% | 0.2982 | **12.91** | **39.4** |
| rescore pending | 52.67% | 19.65% | 63.39% | **0.3000** | 8.35 | 28.0 |
| drop pending | 48.62% | 18.14% | 62.00% | 0.2807 | **5.74** | **16.4** |

No strategy dominates every objective.

## Relative to the frozen incumbent

`retain_old_scores`:

- precision: **+2.27 pp**;
- fraud recall: **+0.85 pp**;
- fraud-value recall: **−3.99 pp**;
- balanced H-mean: **+0.0053**;
- mean delay: **+5.19 steps**;
- p90 delay: **+11.4 steps**.

`rescore_pending`:

- precision: **+1.78 pp**;
- fraud recall: **+0.67 pp**;
- fraud-value recall: **−0.71 pp**;
- balanced H-mean: **+0.0071**;
- mean delay: **+0.63 steps**;
- p90 delay: unchanged at **28 steps**.

`drop_pending`:

- precision: **−2.27 pp**;
- fraud recall: **−0.85 pp**;
- fraud-value recall: **−2.09 pp**;
- balanced H-mean: **−0.0123**;
- mean delay: **−1.98 steps**;
- p90 delay: **−11.6 steps**.

The point estimates therefore do not support a simple statement such as “always rescore” or “always retain old scores”.

## Why retaining old scores changes later capacity

The 617 reviews can be grouped by the arrival cycle of the reviewed transaction:

| Strategy | Cycle-1 arrivals | Cycle-2 arrivals | Cycle-3 arrivals |
|---|---:|---:|---:|
| frozen incumbent | 232 | 325 | 60 |
| retain old scores | **282** | 291 | **44** |
| rescore pending | 236 | 321 | 60 |
| drop pending | 212 | 333 | **72** |

Retaining old scores allows more early-period cases to survive in the backlog and consume later capacity. That raises overall case precision/recall here, but it also reduces the number of late high-fraud arrivals reviewed and lowers fraud-value recall.

Rescoring the backlog largely restores the allocation pattern towards the frozen reference, while drop-at-refresh shifts more capacity towards recent arrivals.

This is a queue-allocation effect, not merely a change in scalar model metrics.

## Handoff volume is itself an operational cost

At step **645** there are **42,204** pending eligible cases.

At step **695**:

- `retain_old_scores` and `rescore_pending` carry **108,643** pending cases;
- `rescore_pending` therefore recomputes **108,643** pending scores at the second refresh;
- `drop_pending` has **66,439** pending cases because the earlier backlog was already expired.

The rescore strategy gives the highest balanced H-mean point estimate, but it does so while requiring a large refresh-time scoring operation. The project does not treat this computational/operations burden as free.

## Queue churn

Relative to the frozen incumbent:

- `retain_old_scores`: **91.25%** overlap, 54 replacements;
- `rescore_pending`: **97.08%** overlap, 18 replacements;
- `drop_pending`: **96.11%** overlap, 24 replacements.

Relative to `retain_old_scores`:

- `rescore_pending`: **92.54%** overlap, 46 replacements;
- `drop_pending`: **88.65%** overlap, 70 replacements.

A refresh can therefore materially change individual investigation assignments even when aggregate metrics move only modestly.

## Interpretation boundaries

This audit is causal with respect to transaction arrival and model-release timing. It is **not** a complete production label-maturity simulation.

The rolling models retain the v1.8 assumption that prior-period labels are available for the next refresh. PaySim does not contain real chargeback maturity, investigation completion or analyst disposition timestamps. A transaction being pending in this routing abstraction therefore does not prove that its upstream fraud label would also be unavailable.

The pending set is an **eligibility backlog**, not a claim that an analyst UI would literally display more than 100,000 unresolved cases. A production system would normally add expiry, candidate-admission or service-level controls.

PaySim transaction amount is not interpreted as prevented loss, and PaySim steps are not translated to hours or days.

## Next methodological boundary

The large pending pool makes the next useful test clear: add an explicit **backlog age / expiry contract** and quantify the quality–latency–backlog-size frontier. That is more informative than adding another classifier.

Candidate policies include fixed maximum pending ages and refresh-aware expiry rules, always with the same causal arrival ordering and analyst-capacity accounting.

## Reproducibility

The dedicated GitHub Actions workflow downloads canonical PaySim, rebuilds strict prior-step DuckDB features, fits the three declared rolling systems, runs all four continuous handoff strategies, uploads the artifact and writes aggregate results back to `main` after merge.

Primary outputs:

- `results/paysim_backlog_handoff/strategy_metrics.csv`
- `results/paysim_backlog_handoff/arrival_cohort_metrics.csv`
- `results/paysim_backlog_handoff/refresh_handoff_diagnostics.csv`
- `results/paysim_backlog_handoff/strategy_queue_overlap.csv`
- `results/paysim_backlog_handoff/reviews_by_regime.csv`
- `results/paysim_backlog_handoff/continuous_capacity_schedule.csv`
- `results/paysim_backlog_handoff/summary.json`
