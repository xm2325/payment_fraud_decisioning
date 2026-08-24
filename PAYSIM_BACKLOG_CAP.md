# PaySim bounded priority backlog frontier

## Why this audit exists

v1.13 showed that a continuous causal review backlog creates a large model-release workload: the unbounded `rescore_pending` policy rescored **42,204** pending cases at step 645 and **108,643** at step 695, or **150,847** refresh rescoring operations in total.

v1.14 tested a label-free age TTL. No finite TTL simultaneously passed the frozen detection guardrails and the >=50% refresh-workload reduction target. Age expiry was therefore retained as a negative result.

v1.15 tests a different, predeclared operations mechanism: **bound the pending pool by current policy priority instead of age**. After allocating newly earned analyst capacity, if the pending pool exceeds a fixed cap, the lowest current-score cases are evicted. At a model refresh, every surviving pending case is rescored by the newly released system.

The intent is not to find another predictive model. It is to ask whether a hard release-workload bound can be imposed while preserving the realised investigation queue.

## Frozen experiment

Before reading full-data results, the cap grid was fixed as:

`max_pending_cases ∈ {0, 5,000, 10,000, 25,000, 50,000, 100,000, infinite}`.

Every row uses:

- the same canonical **6,362,620-row PaySim** source;
- the same three rolling model/calibration/policy regimes;
- the same validation-selected `balanced` routing profile;
- one continuous causal horizon over steps **595--743**;
- the same exact capacity of **50 reviews per 10,000 arrivals = 617 reviews**;
- rescore-on-refresh for all retained pending cases;
- stable non-label event-key tie-breaking.

Newly earned analyst capacity is allocated before the memory cap is enforced. If the remaining pending pool exceeds the cap, the lowest current score is evicted. For equal scores, the larger event key is evicted first, preserving the same deterministic priority order used for review selection.

Fraud labels are never used to evict or select cases.

`max_pending_cases=infinite` is a reproducibility anchor and reproduces the v1.13 `rescore_pending` result exactly.

## Frozen operational screen

The v1.14 screen is reused unchanged. Relative to the infinite backlog, a bounded row must satisfy all of:

- precision decline no worse than **2 percentage points**;
- fraud-recall decline no worse than **2 percentage points**;
- fraud-value-recall decline no worse than **2 percentage points**;
- total refresh rescoring reduced by at least **50%**.

`operational_candidate` is a deterministic engineering screen, not a confidence-bound promotion test and not a production recommendation.

## Full-data frontier

All rows consume exactly 617 reviews.

| Pending cap | Precision | Fraud recall | Fraud-value recall | H-mean | Mean delay | Refresh rescores | Reduction vs ∞ | Queue overlap vs ∞ | Candidate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 23.34% | 8.71% | 27.33% | 0.1321 | 0.00 | 0 | 100.00% | 48.46% | no |
| **5,000** | **52.67%** | **19.65%** | **63.39%** | **0.3000** | **8.35** | **10,000** | **93.37%** | **100.00%** | **yes** |
| 10,000 | 52.67% | 19.65% | 63.39% | 0.3000 | 8.35 | 20,000 | 86.74% | 100.00% | yes |
| 25,000 | 52.67% | 19.65% | 63.39% | 0.3000 | 8.35 | 50,000 | 66.85% | 100.00% | yes |
| 50,000 | 52.67% | 19.65% | 63.39% | 0.3000 | 8.35 | 92,204 | 38.88% | 100.00% | no: workload |
| 100,000 | 52.67% | 19.65% | 63.39% | 0.3000 | 8.35 | 142,204 | 5.73% | 100.00% | no: workload |
| ∞ | 52.67% | 19.65% | 63.39% | 0.3000 | 8.35 | 150,847 | 0% | 100.00% | reference |

The surprising result is not merely that the detection metrics round to the same values. For every tested non-zero cap, the realised 617-review set is **exactly identical** to the infinite-backlog review set. For the 5k, 10k and 25k caps:

- 617/617 reviews overlap;
- replacement count = **0**;
- precision delta = **0**;
- fraud-recall delta = **0**;
- fraud-value-recall delta = **0**;
- aggregate review-delay metrics are also unchanged.

Therefore 5k, 10k and 25k all pass the unchanged v1.14 screen.

## Why the 5k row is notable

5,000 is the **lowest non-zero cap in the predeclared grid**, not a post-hoc estimate of the true minimum sufficient backlog.

At each release it rescored exactly 5,000 retained pending cases:

- step 645: **5,000** instead of 42,204;
- step 695: **5,000** instead of 108,643.

Total refresh rescoring is therefore **10,000 instead of 150,847**, a **93.37% reduction**, while the realised investigation queue remains exactly unchanged on this benchmark/horizon.

The project does **not** infer that caps below 5,000 would also work. Testing a denser sub-5k grid after observing this result would be a new experiment and would need its own predeclared validation plan.

## Why score-based capping differs from TTL expiry

The age TTL in v1.14 discards a case because it is old, regardless of whether it still has high current policy priority. That created a direct quality/workload trade-off: TTL20 met the workload target but missed the precision guardrail.

The priority cap asks a different question: **which pending cases are worth retaining at all under a fixed analyst budget?** It evicts the ranking tail and keeps the highest current-priority cases available for future capacity and refresh-time rescoring.

In this realised horizon, a 5k retained reservoir is sufficient to contain every case that ultimately appears in the infinite-backlog 617-review set.

This is evidence about the realised queue under this benchmark. It is not proof that 5,000 is universally sufficient under different fraud regimes, capacities or score drift.

## Do not misread the eviction counts

The 5k cap eventually marks **117,963** transactions as evicted, including 827 labelled fraud cases in retrospective diagnostics. That does **not** mean the cap caused 827 missed investigations.

Under the infinite baseline, only 617 of 123,580 future transactions are ever reviewed. The bounded 5k policy reviews the **same 617 transactions** as the infinite policy, including the same 325 fraud cases and the same captured fraud value.

The evicted-label counts describe the composition of the low-priority tail that was removed from future eligibility. They are useful for audit, but the direct causal comparison of realised assignments is the queue-overlap table.

## What this adds to the portfolio story

The sequence is now explicit:

1. whole-window top-k was corrected because it used future-score hindsight;
2. a causal seen-so-far backlog recovered much of the ranking quality;
3. cross-refresh handoff exposed a 150,847-case rescoring workload;
4. age TTL could not satisfy frozen detection + workload requirements;
5. a score-priority memory cap did satisfy the same requirements without changing the realised 617-review queue for the tested 5k/10k/25k caps.

This is a decision-systems result rather than another classifier benchmark.

## Claim boundaries

- PaySim is synthetic mobile-money data, not Moniepoint production data.
- PaySim steps are dataset time units, not measured analyst SLA units.
- Transaction amount is not prevented loss or realised business impact.
- Labels are used only for retrospective evaluation; they do not drive memory eviction, scoring or review selection.
- The 2 pp and 50% thresholds are deterministic engineering guardrails, not statistical confidence bounds.
- Passing `operational_candidate` does not establish production readiness.
- The rolling lifecycle retains the existing as-of assumption that prior-period labels are available at the next refresh; PaySim has no real investigation-completion or label-maturity timestamps.
- 5,000 is the lowest **tested** non-zero cap in the predeclared grid. Do not claim it is the minimum required cap.
