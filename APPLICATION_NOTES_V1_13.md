# Application notes — v1.13 backlog handoff evidence

Use this evidence when a role stresses fraud operations, model deployment, queue management, production ML lifecycle, monitoring or governance.

PaySim is synthetic. Do not present these figures as production impact, prevented loss, staffing evidence or a real service-level agreement.

## Preferred deployment-lifecycle bullet

Built a continuous fraud-review queue audit on the full **6.36M-row PaySim** benchmark, testing how unresolved cases should cross rolling model releases under the same **617-review** capacity; rescoring pending cases gave the best balanced case/value metric (**H-mean 0.3000**) but required rescoring **42k then 109k** eligible cases, while retaining old scores increased mean review delay from **7.7 to 12.9 PaySim steps** and reduced fraud-value recall by **4.0 pp**.

This is stronger than saying “I retrained the model periodically” because it tests what happens to already-scored cases at the release boundary.

## Shorter version

Audited fraud-queue handoff across model refreshes on **6.36M PaySim transactions**; compared frozen, retain-score, rescore and expiry policies at identical analyst capacity, exposing trade-offs between recall, fraud-value capture, review delay, queue churn and **100k+ pending-case** rescore volume.

## Interview story

### 1. A model refresh creates an operations decision

The project already had rolling model/calibration/policy refreshes. v1.12 evaluated each later test window causally, but each window started from an empty evaluation backlog.

That is convenient statistically, but incomplete operationally. When a model is released, unresolved cases do not automatically disappear.

I therefore ran one continuous future horizon from PaySim step **595 through 743** and kept analyst capacity continuous across the two refresh boundaries at **645** and **695**.

### 2. I held final analyst capacity constant

The full horizon contains **123,580 transactions**. At **50 reviews per 10,000**, the cumulative entitlement is **617 reviews**.

Every handoff strategy uses exactly 617 reviews. This matters because the three individual rolling test windows would floor separately to 212 + 333 + 71 = 616 reviews. I did not let one handoff policy look better by silently giving it more capacity.

### 3. I compared four operational choices

`frozen_incumbent`: never release the later rolling systems.

`retain_old_scores`: release the new system for new arrivals, but preserve pending scores.

`rescore_pending`: release the new system and recompute every pending case score at the boundary.

`drop_pending`: expire unresolved cases at refresh but keep the continuous capacity entitlement.

### 4. No handoff strategy dominates

At 50 reviews/10k:

| Strategy | Precision | Fraud recall | Fraud-value recall | Balanced H-mean | Mean delay |
|---|---:|---:|---:|---:|---:|
| frozen | 50.89% | 18.98% | **64.10%** | 0.2929 | 7.72 |
| retain | **53.16%** | **19.83%** | 60.10% | 0.2982 | **12.91** |
| rescore | 52.67% | 19.65% | 63.39% | **0.3000** | 8.35 |
| expire | 48.62% | 18.14% | 62.00% | 0.2807 | **5.74** |

A good explanation is:

> “Rescoring had the best balanced point metric, but it did not improve every objective: fraud-value recall was still slightly below the frozen queue and review delay increased. Retaining old scores improved case precision and recall, but it kept old cases alive much longer and lost about four percentage points of fraud-value recall. I therefore treated handoff as an operations trade-off, not a model leaderboard.”

### 5. Retaining stale scores can starve recent arrivals

The same 617 reviews are allocated across arrival cohorts very differently:

- frozen: **232 / 325 / 60** reviews from cycles 1 / 2 / 3;
- retain old scores: **282 / 291 / 44**;
- rescore pending: **236 / 321 / 60**;
- drop pending: **212 / 333 / 72**.

Retaining old scores lets more early cases continue consuming later capacity. In this benchmark that raises case-level precision/recall but reduces late-period coverage and fraud-value recall.

This is a useful example of why an apparently harmless deployment rule can change who gets investigated.

### 6. Rescoring has a volume cost

At the first refresh, **42,204** eligible cases remain pending. At the second refresh, the carry-over pool reaches **108,643** cases.

A `rescore_pending` policy therefore requires a large batch scoring operation at release time. The project reports that volume explicitly instead of treating model refresh as free.

### 7. Queue churn is measurable

Compared with the frozen queue:

- retain old scores: **91.25%** overlap;
- rescore pending: **97.08%** overlap;
- drop pending: **96.11%** overlap.

Even a few percentage points of queue change can represent dozens of different investigations at the same analyst capacity.

### 8. The next question is backlog age, not another classifier

The eligibility backlog can become very large because every unreviewed transaction remains eligible indefinitely. A real system would normally add expiry, service-level or candidate-admission rules.

The next sensible audit is a fixed-age/expiry sensitivity that measures:

- precision / fraud recall / fraud-value recall;
- review delay;
- maximum and average eligible backlog size;
- rescore volume at refresh;
- queue churn.

That would turn the handoff analysis into a quality–latency–backlog-size frontier.

## Claims to avoid

Do not say:

- `rescore_pending` is proven to be the production policy;
- the 108,643 pending cases represent a real fraud operations team backlog;
- PaySim steps are hours or days;
- the rolling labels reproduce real chargeback maturity;
- transaction amount equals prevented fraud loss;
- model refresh should always rescore all old cases.

The safe conclusion is: **under one continuous causal PaySim capacity contract, model-refresh handoff rules materially change queue composition, temporal allocation, review delay and case/value trade-offs even when the predictive model family is unchanged.**
