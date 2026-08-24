# PaySim policy-promotion dependence sensitivity — v1.10

## Question

v1.9 uses a paired circular **5-step block bootstrap** to quantify uncertainty when comparing the frozen v1.7 routing policy with a rolling-refresh candidate. The strongest candidate was cycle-3 `value_first`: fraud-value recall increased by **20.20 percentage points**, but the 5-step fraud-case-recall lower bound was **-2.98 pp**, outside the pre-declared **-2 pp** non-inferiority margin, so the candidate stayed `KEEP_INCUMBENT`.

v1.10 asks a narrower robustness question: **does that governance conclusion depend on the assumed temporal block length?**

This audit does not replace, tune, or reinterpret the frozen v1.9 gate. The official v1.9 decision remains based on 5-step blocks.

## Frozen sensitivity contract

The sensitivity set is declared as block lengths **1, 3, 5 and 10 steps**. For every block length, the audit reruns the same six comparisons:

- cycles 2 and 3;
- `case_first`, `balanced`, and `value_first` profiles;
- exact capacity of 50 reviews per 10,000 transactions;
- 2,000 paired circular block-bootstrap replicates per comparison;
- family-wise alpha 0.05 over six comparisons, giving one-sided Bonferroni tail alpha 0.0083333;
- minimum primary gain +2 pp;
- non-primary non-inferiority margins of -2 pp.

The 5-step sensitivity rows must numerically reproduce the committed v1.9 `policy_promotion_gate.csv`. The full-data workflow passed this check.

## Full 6.36M-row result

Five of the six cycle/profile comparisons are `ROBUST_KEEP_INCUMBENT`: every declared block length returns `KEEP_INCUMBENT`.

The only dependence-sensitive comparison is **cycle 3 / value-first**:

| Block length | Value-recall point gain | Family-adjusted value-recall LCB | Fraud-recall LCB | Decision |
|---:|---:|---:|---:|---|
| 1 step | +20.20 pp | +5.93 pp | -3.74 pp | KEEP_INCUMBENT |
| 3 steps | +20.20 pp | +8.12 pp | -3.13 pp | KEEP_INCUMBENT |
| 5 steps | +20.20 pp | +9.78 pp | -2.98 pp | KEEP_INCUMBENT |
| 10 steps | +20.20 pp | +12.39 pp | -1.84 pp | PROMOTE |

The primary value-recall gain remains positive under every declared dependence assumption. What changes is the case-recall guardrail: at 1, 3, and 5 steps its lower bound is below -2 pp, while at 10 steps it rises to -1.84 pp and passes.

The sensitivity classification is therefore **`DEPENDENCE_SENSITIVE`**, not `ROBUST_PROMOTE`.

## Interpretation

The aggregate cycle-3 fraud-case recall is unchanged between the incumbent and candidate, but time-block resampling reveals uncertainty in where those captured cases occur across the test period. The decision can therefore change when the assumed dependence scale changes even though the full-window point estimate does not.

This is exactly why v1.10 does not choose the 10-step result after seeing that it promotes the candidate. A post-hoc block-length choice would turn a robustness analysis into parameter shopping. The frozen v1.9 5-step decision remains `KEEP_INCUMBENT`.

The operational conclusion is:

> The candidate has strong evidence of higher captured fraud value, but the release decision is not robust to plausible temporal-dependence assumptions. Keep the incumbent and collect more completed periods before reconsidering promotion.

## Reproducibility

The dedicated GitHub Actions workflow downloads the canonical PaySim parquet shards, validates **6,362,620 transactions / 8,213 frauds / steps 1-743**, rebuilds point-in-time features, reruns the rolling lifecycle, checks exact v1.9 5-step reproduction, runs all four dependence assumptions, writes `block_sensitivity.csv`, `sensitivity_summary.csv`, and `summary.json`, and uploads the aggregate result artifact.

On `main` pushes the workflow commits these aggregate results to `results/paysim_promotion_sensitivity/`; raw PaySim rows are never committed.

## Boundaries

These intervals quantify temporal uncertainty in realised outcomes for two frozen queues. They do **not** include model-refit uncertainty, future prevalence uncertainty, intervention effects, analyst behaviour, delayed investigation completion, or real label-maturity timing.

PaySim is synthetic mobile-money data. The result is portfolio benchmark evidence, not production impact or prevented-loss evidence.
