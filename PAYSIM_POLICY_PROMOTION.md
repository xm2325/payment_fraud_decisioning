# PaySim policy promotion gate — v1.9

## Question

v1.8 found a large late-period point estimate for the value-first routing profile: in cycle 3, the refreshed policy selected `alpha=1.0` and increased fraud-value recall from **40.76% to 60.95%** at 50 reviews per 10,000 transactions. A point estimate alone is not enough to replace an incumbent policy.

v1.9 asks a narrower release-governance question:

> After a completed evaluation window has labels, is there enough paired, time-aware evidence to promote a refreshed routing policy over the frozen v1.7 incumbent?

The answer on canonical PaySim is **no for all six pre-declared cycle/profile comparisons**. The strongest candidate passes the primary value-recall test but fails a case-recall non-inferiority guardrail.

## Frozen comparison family

The comparison family is fixed before reading the cycle-2/3 test outcomes:

- cycle 2: `case_first`, `balanced`, `value_first`;
- cycle 3: `case_first`, `balanced`, `value_first`.

That gives **6 comparisons**. A refreshed policy is treated as a distinct candidate even when its selected `alpha` equals the incumbent `alpha`, because the fitted model, calibrator, amount scale and therefore queue ordering may still differ.

The incumbent is the frozen v1.7 model/calibrator/routing bundle from cycle 1. The candidate is the rolling-refresh bundle selected using only data before its test window.

## Exact-capacity policy comparison

Both incumbent and candidate use the same operational budget:

**50 reviews per 10,000 transactions**.

Each policy first creates its deterministic exact top-k queue using score descending and the existing non-label `event_key` for ties. Test labels do not affect queue construction.

For uncertainty, v1.9 keeps those two realised queues fixed and applies a **paired circular moving-block bootstrap over time steps**. The bootstrap therefore compares incremental captured outcomes under the two policies while preserving short-range temporal dependence. It does not refit the model inside each bootstrap replicate.

Configuration:

| Item | Value |
|---|---:|
| Bootstrap replicates | 2,000 |
| Circular block length | 5 PaySim steps |
| Family alpha | 0.05 |
| Number of comparisons | 6 |
| Bonferroni one-sided tail alpha | 0.008333 |
| Minimum primary gain | +2 percentage points |
| Precision non-inferiority margin | -2 pp |
| Case-recall non-inferiority margin | -2 pp |
| Value-recall non-inferiority margin | -2 pp |

The family-adjusted lower bound is therefore the 0.8333rd percentile of the paired bootstrap delta distribution for each comparison.

## Profile-specific primary metrics

The policy objectives remain explicit:

- `case_first`: fraud-case recall;
- `balanced`: harmonic mean of fraud-case recall and fraud-value recall;
- `value_first`: fraud-value recall.

A candidate is promoted only if all of the following hold:

1. the primary point estimate improves by at least **2 pp**;
2. the family-adjusted one-sided lower bound for the primary delta is **strictly positive**;
3. precision and any non-primary recall metric remain within their declared **-2 pp non-inferiority margins**.

The rule is fixed before interpreting the full-data results.

## Full 6.36M-row result

### Cycle 2

No cycle-2 candidate has a meaningful primary gain.

| Profile | Incumbent alpha | Candidate alpha | Primary metric | Point delta | Family-adjusted lower bound | Decision |
|---|---:|---:|---|---:|---:|---|
| case-first | 0.25 | 0.25 | case recall | 0.00 pp | -0.58 pp | KEEP_INCUMBENT |
| balanced | 0.25 | 0.25 | balanced H-mean | -0.02 pp | -0.81 pp | KEEP_INCUMBENT |
| value-first | 0.25 | 0.50 | value recall | -0.21 pp | -0.54 pp | KEEP_INCUMBENT |

The value-first cycle-2 candidate also slightly reduces case recall and precision. There is no basis for promotion.

### Cycle 3

The late value-first result is the important case.

| Profile | Incumbent alpha | Candidate alpha | Primary metric | Point delta | Family-adjusted lower bound | Decision |
|---|---:|---:|---|---:|---:|---|
| case-first | 0.25 | 0.25 | case recall | 0.00 pp | 0.00 pp | KEEP_INCUMBENT |
| balanced | 0.25 | 0.25 | balanced H-mean | 0.00 pp | 0.00 pp | KEEP_INCUMBENT |
| value-first | 0.25 | 1.00 | value recall | **+20.20 pp** | **+9.78 pp** | **KEEP_INCUMBENT** |

The value-first primary metric is strong: the family-adjusted lower bound remains almost **+10 pp**, far above zero. However, the paired time-block bootstrap gives a fraud-case recall lower bound of **-2.98 pp**. The pre-declared guardrail allows only **-2 pp**. Therefore the candidate does not pass release promotion.

This is deliberately stricter than reading the full-window point estimates, where incumbent and candidate both have **12.77% case recall** and **100% precision**. The full-window totals hide the fact that the two queues capture different fraud cases at different times. Resampling contiguous time blocks exposes that temporal instability.

## Interpretation

v1.9 changes the project conclusion in an important way.

The correct statement is not:

> “Cycle-3 value-first improved fraud-value recall by 20.2 pp, so switch to alpha=1.0.”

The evidence supports:

> “Cycle-3 value-first produced a large and statistically stable fraud-value gain under a paired time-block bootstrap, but it failed the pre-declared case-recall non-inferiority guardrail. I therefore kept the incumbent policy.”

This demonstrates a release decision that can reject an attractive headline result when a second operational objective is not sufficiently protected.

## What this uncertainty does and does not mean

The bootstrap measures temporal uncertainty in **incremental realised queue outcomes** after both policies are frozen for the completed test window. It is not a confidence interval for model parameters, it does not include retraining uncertainty, and it does not guarantee future performance.

The promotion decision is retrospective. Test labels may be used only after the test window is completed and labels are considered available; the decision cannot change or claim the same window's performance.

PaySim itself has no real fraud investigation-completion or chargeback-maturity timestamps. The project therefore does not claim that these release timings are production-valid. Explicit delayed-label behaviour remains covered by the controlled 120k stress-test layer.

## Reproducibility

The implementation is in:

- `src/fraud_decisioning/paysim_policy_promotion.py`;
- `scripts/run_paysim_rolling_refresh.py`;
- `tests/test_paysim_policy_promotion.py`.

Full-data aggregate evidence is written by GitHub Actions to:

- `results/paysim_rolling_refresh/policy_promotion_gate.csv`;
- `results/paysim_rolling_refresh/summary.json`.

Raw PaySim transactions are not committed.
