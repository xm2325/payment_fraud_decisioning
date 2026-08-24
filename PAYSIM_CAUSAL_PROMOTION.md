# PaySim causal promotion audit — v1.12

## Why this audit exists

v1.9 and v1.10 added a governed candidate-versus-incumbent release gate on the rolling PaySim lifecycle. Those audits were careful about time-separated model fitting, probability calibration, policy selection, family-wise error control and time-block uncertainty, but the realised test queues still used a whole-window exact top-k rule.

v1.11 showed that whole-window top-k is label-safe but not an online routing contract: an earlier review decision can depend on scores from later PaySim steps. The old numbers therefore remain useful as **retrospective completed-window benchmarks**, not as causal Fraud Ops results.

v1.12 changes only that evaluation layer. It keeps the existing governance thresholds and rebuilds incumbent and candidate test queues with the **seen-so-far backlog** contract.

## Frozen governance contract

The following are not re-tuned after seeing the causal result:

- exact capacity: **50 reviews per 10,000 transactions**;
- six family comparisons: cycles 2–3 × `case_first`, `balanced`, `value_first`;
- minimum primary gain: **+2 percentage points**;
- precision, fraud-recall and fraud-value-recall non-inferiority margins: **−2 percentage points** when they are guardrails rather than the primary objective;
- family alpha: **0.05**, using a Bonferroni one-sided tail alpha of **0.0083333** per comparison;
- official time-block length: **5 PaySim steps**;
- bootstrap replicates: **2,000**;
- the official 5-step bootstrap uses the same deterministic seed rule as v1.9.

The 1/3/5/10-step block set remains a sensitivity audit. It cannot replace the 5-step official gate after seeing a favourable result.

## Rolling lifecycle

| Cycle | Training | Calibration | Policy selection | Test |
|---:|---|---|---|---|
| 1 | 1–445 | 446–519 | 520–594 | 595–644 |
| 2 | 1–495 | 496–569 | 570–644 | 645–694 |
| 3 | 1–545 | 546–619 | 620–694 | 695–743 |

The incumbent remains the frozen cycle-1 v1.7 system. The candidate is refit/recalibrated/reselected from the declared prior stages for each later cycle.

Completed policy-selection windows can use their full historical scores to choose routing alpha because they precede the test period. Within a test period, the causal backlog uses only transactions that have arrived by the current step.

## Official 5-step causal gate

All six comparisons remain `KEEP_INCUMBENT`.

| Cycle | Profile | Incumbent α | Candidate α | Primary point Δ | Family-adjusted lower bound | Decision |
|---:|---|---:|---:|---:|---:|---|
| 2 | case-first | 0.25 | 0.25 | case recall **0.00 pp** | **−0.58 pp** | KEEP_INCUMBENT |
| 2 | balanced | 0.25 | 0.25 | H-mean **−0.02 pp** | **−0.81 pp** | KEEP_INCUMBENT |
| 2 | value-first | 0.25 | 0.50 | value recall **−0.21 pp** | **−0.54 pp** | KEEP_INCUMBENT |
| 3 | case-first | 0.25 | 0.25 | case recall **−0.18 pp** | **−0.94 pp** | KEEP_INCUMBENT |
| 3 | balanced | 0.25 | 0.25 | H-mean **−0.43 pp** | **−1.59 pp** | KEEP_INCUMBENT |
| 3 | value-first | 0.25 | 1.00 | value recall **+0.95 pp** | **−0.81 pp** | KEEP_INCUMBENT |

## The cycle-3 value-first result changes materially

The retrospective whole-window result made the cycle-3 value-first candidate look large:

- fraud-value-recall point delta: **+20.20 pp**;
- family-adjusted lower bound: **+9.78 pp**;
- v1.10 dependence sensitivity: `KEEP, KEEP, KEEP, PROMOTE` across 1/3/5/10-step blocks.

Under the causal seen-so-far backlog:

- incumbent fraud-value recall: **39.46%**;
- candidate fraud-value recall: **40.41%**;
- point delta: **+0.95 pp**;
- family-adjusted 5-step lower bound: **−0.81 pp**;
- incumbent precision: **87.32%**;
- candidate precision: **84.51%**;
- precision point delta: **−2.82 pp**;
- precision lower bound: **−7.47 pp**;
- fraud-recall point delta: **−0.36 pp**;
- fraud-recall lower bound: **−1.70 pp**.

So the causal candidate fails promotion for several independent reasons: the primary gain is below the pre-declared +2 pp minimum, its lower bound is not positive, and its precision guardrail does not pass.

The causal incumbent/candidate queues overlap by **91.55%** in this comparison, with six of 71 review assignments replaced. The candidate mean review delay is **2.68 PaySim steps** and p90 is **8 steps** within this test window.

## Dependence sensitivity becomes simpler, not more favourable

Under causal backlog routing, **all six cycle/profile comparisons are `ROBUST_KEEP_INCUMBENT` across 1/3/5/10-step blocks**.

The main qualitative change from v1.10 is cycle-3 value-first:

- retrospective: `DEPENDENCE_SENSITIVE`, with 10-step blocks reaching `PROMOTE`;
- causal backlog: `ROBUST_KEEP_INCUMBENT`, with all four block lengths returning `KEEP_INCUMBENT`.

This is stronger evidence against promotion than the earlier retrospective result. It also shows why a governance layer cannot repair an unrealistic queue contract underneath it: first make the operational evaluation causal, then quantify uncertainty.

## What did not change

Cycle-2 results are nearly identical between retrospective and causal evaluation because incumbent and candidate queues are almost the same in that period. At 50 reviews/10k, the causal queue overlap is **99.70%** for case-first/balanced and **99.40%** for value-first.

Cycle-3 case-first/balanced are no longer numerically identical after causal routing even though both incumbent and candidate use `alpha=0.25`, because the rolling model/calibrator changes the score ordering. Their queue overlap is **94.37%**.

## Uncertainty scope

The causal queues are constructed once on the observed test sequence. The paired circular block bootstrap then resamples **realised per-step outcomes** for the two frozen queues. This matches the scope of v1.9: it measures temporal uncertainty in realised incremental outcomes.

It does **not** include:

- model-refit uncertainty;
- calibrator-refit uncertainty;
- alternative queue reconstruction under every bootstrap history;
- real label-maturity or investigation-completion uncertainty.

Those exclusions are explicit rather than folded into a broad statistical claim.

## Boundary still open after v1.12

Each rolling test window starts with an empty evaluation backlog. This matches the existing cycle-by-cycle promotion design, but it is not a full continuous deployment simulation across model-refresh boundaries.

A production design would need to decide what happens to pending cases when a new model is released: retain old scores, rescore the backlog, expire old cases, or apply a mixed hand-off policy. v1.12 does not claim to resolve that lifecycle decision.

## Reproducibility

The dedicated GitHub Actions workflow downloads the canonical PaySim parquet shards, verifies **6,362,620 transactions / 8,213 fraud cases / steps 1–743**, rebuilds strict prior-step DuckDB features, reruns the rolling models and causal queues, executes the 24 block-sensitivity comparisons, uploads the artifact and writes aggregate results back to `main` after merge.

Primary outputs:

- `results/paysim_causal_promotion/causal_promotion_gate.csv`
- `results/paysim_causal_promotion/causal_block_sensitivity.csv`
- `results/paysim_causal_promotion/causal_sensitivity_summary.csv`
- `results/paysim_causal_promotion/retrospective_vs_causal_promotion.csv`
- `results/paysim_causal_promotion/retrospective_vs_causal_sensitivity.csv`
- `results/paysim_causal_promotion/summary.json`

PaySim is synthetic. Transaction amount is not interpreted as prevented loss, and PaySim steps are not interpreted as a production service-level unit.
