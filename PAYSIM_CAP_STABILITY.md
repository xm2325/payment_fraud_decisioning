# PaySim 5k backlog-cap pathwise stability audit (v1.16)

## Question

v1.15 showed that a 5,000-case score-priority pending backlog produced the same final 617-review queue as an infinite backlog at 50 reviews per 10k while cutting refresh-time rescoring by 93.37%.

That aggregate result does not by itself prove the queue evolved identically through time. A bounded queue could, in principle, review the same final cases at different steps, or preserve the 50/10k result but diverge when analyst capacity changes.

v1.16 therefore freezes the 5,000-case cap and asks a stricter question: **does it preserve the realised review path under predeclared analyst-capacity stress?**

## Frozen design

No model, alpha, cap or routing profile is selected from the v1.16 results.

- canonical PaySim: 6,362,620 transactions / 8,213 fraud cases;
- same v1.8 rolling train/calibration/policy/test lifecycle;
- same balanced profile selected on policy windows at the original 50 reviews/10k policy-selection capacity;
- same causal rescore-on-refresh handoff;
- candidate pending cap: **5,000**, fixed from v1.15;
- no sub-5,000 search;
- capacity stress grid fixed before reading results: **10, 25, 50, 100 reviews per 10k**;
- infinite backlog is the comparator separately at each capacity.

`PATHWISE_PRESERVED` is deliberately stronger than matching aggregate fraud metrics. It requires:

1. the same final selected mask; and
2. the same review step for every reviewed case.

The audit also records the first divergent review step, count of divergent review steps, minimum cumulative queue Jaccard, arrival-cohort displacement and refresh-time rescore volume.

## Full canonical PaySim result

| Capacity / 10k | Reviews | Final overlap | Same review timing | Divergent review steps | Minimum cumulative Jaccard | Infinite refresh rescores | 5k refresh rescores | Reduction |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10 | 123 | 123 / 123 | yes | 0 | 1.000 | 151,453 | 10,000 | 93.40% |
| 25 | 308 | 308 / 308 | yes | 0 | 1.000 | 151,226 | 10,000 | 93.39% |
| 50 | 617 | 617 / 617 | yes | 0 | 1.000 | 150,847 | 10,000 | 93.37% |
| 100 | 1,235 | 1,235 / 1,235 | yes | 0 | 1.000 | 150,089 | 10,000 | 93.34% |

All four predeclared capacities are `PATHWISE_PRESERVED`.

This means the 5,000-case backlog did not merely end with the same reviewed cases. Across steps 595-743 it selected the same case at the same review step as the infinite backlog for every realised review at every tested capacity.

The existing three rolling arrival cohorts also show zero final displacement at every tested capacity. For example, at 100 reviews/10k the 1,235 reviews split 435 / 689 / 111 across the three arrival cohorts for both 5k and infinite queues.

## Reproduction gate

The PR-context full-data workflow reads the committed v1.15 `results/paysim_backlog_cap/cap_frontier.csv` reference. At 50 reviews/10k, both the infinite and 5,000 rows reproduce the v1.15 alerts, precision, fraud recall, fraud-value recall, H-mean, review-delay metrics and refresh-rescore totals to tight numerical tolerance.

`v1_15_capacity50_reference_reproduced = true`.

## What this does and does not establish

This is strong **queue-mechanics evidence** for this synthetic benchmark and fixed rolling policy. It shows that retaining only the top 5,000 currently scored pending cases was sufficient to preserve the complete realised investigation path across a four-point capacity stress grid while sharply limiting refresh-time rescoring.

It does **not** establish that:

- 5,000 is the true minimum sufficient pending pool;
- a smaller cap would fail (v1.16 deliberately does not search below 5,000);
- any tested analyst capacity matches a real production operation;
- the same cap would preserve a queue under different traffic, prevalence, scoring distributions or label-maturity processes;
- PaySim transaction amount is prevented loss;
- PaySim steps correspond to a production SLA unit.

The correct interpretation is therefore: **the predeclared 5,000-case score-priority reservoir preserved the exact realised review path across 10/25/50/100 reviews per 10k on this canonical PaySim rolling benchmark, while reducing refresh rescoring by about 93.3%-93.4%.**
