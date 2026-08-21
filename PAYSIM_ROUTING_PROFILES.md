# PaySim routing-governance audit — v1.5

v1.5 asks a narrower operational question than model selection: **given a fixed Fraud Ops review capacity, how should transactions be prioritised when case coverage and fraud-value coverage are both important?**

The predictive model is unchanged from v1.4 (`transaction_plus_relational`). The new object is a routing score:

`priority = calibrated P(fraud) × (amount / validation_amount_scale)^alpha`

with a pre-specified grid `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`. `alpha=0` is pure risk ranking and `alpha=1` is expected-loss-style `P(fraud) × amount`. The amount scale is fitted on validation only and does not change ordering for a fixed alpha.

## Selection contract

The routing exponent is not tuned on future test data. At exactly **50 alerts per 10,000 transactions** on validation, three pre-declared objectives are evaluated:

- case-first: maximise fraud-case recall;
- balanced: maximise the harmonic mean of case recall and fraud-value recall;
- value-first: maximise fraud-value recall.

All three objectives independently select **alpha=0.25**. Because the three profiles collapse to the same policy, the portfolio treats this as one **validation-selected compromise ranker**, not three operational profiles.

### Validation evidence at 50 alerts / 10k

| alpha | interpretation | precision | fraud recall | fraud-value recall |
|---:|---|---:|---:|---:|
| 0.00 | probability only | 38.16% | 28.03% | 74.43% |
| **0.25** | **validation-selected compromise** | **38.51%** | **28.29%** | **79.30%** |
| 0.50 | more amount weight | 36.84% | 27.06% | 78.55% |
| 0.75 | more amount weight | 36.84% | 27.06% | 78.55% |
| 1.00 | probability × amount | 36.84% | 27.06% | 78.55% |

The important point is not that 0.25 is universally optimal. It is that a modest amount weight improves both validation case recall and validation value recall relative to the two endpoint policies on this benchmark, so the choice can be frozen before future evaluation.

## Untouched future result

At the same **50 alerts / 10k** capacity, the frozen `alpha=0.25` ranker gives:

- precision **61.75%**;
- fraud-case recall **23.04%**;
- fraud-value recall **77.70%**;
- 19.36 legitimate alerts per 10,000 legitimate transactions.

For context, the already-verified v1.4 endpoints at exactly the same capacity are:

| ranker | precision | fraud recall | fraud-value recall |
|---|---:|---:|---:|
| probability only (`alpha=0`) | **64.34%** | **24.00%** | 71.67% |
| **validation-selected `alpha=0.25`** | 61.75% | 23.04% | **77.70%** |
| probability × amount (`alpha=1`) | 56.40% | 21.04% | 76.96% |
| amount/type rule | 42.46% | 15.84% | 70.43% |

Relative to probability-only ranking, `alpha=0.25` sacrifices about **2.59 percentage points of precision** and **0.97 points of case recall** for about **6.03 points of fraud-value recall**. Relative to `alpha=1`, the selected compromise improves all three reported metrics on this future period.

## Temporal robustness boundary

The validation-selected compromise does **not** solve regime shift. At 50 alerts / 10k, fraud-value recall across the three future windows is approximately:

- window 1: **75.45%**;
- window 2: **82.30%**;
- window 3: **40.76%**.

The final window remains capacity-constrained during a high-fraud period. v1.5 therefore does not claim that a static alpha makes the queue robust; it shows how to select a routing objective without future-test tuning and then expose where that frozen policy fails.

## Evidence boundary

- Alpha values were pre-specified before future evaluation.
- Selection uses validation labels only; the future alpha grid is sensitivity-only and cannot be used to re-select alpha.
- All comparisons use exact top-k review capacity and the same stable non-label event-key tie-breaker.
- `P(fraud) × amount^alpha` is a prioritisation heuristic, not a prevented-loss estimate.
- PaySim is synthetic mobile-money data. These are engineering and decision-policy benchmark results, not production performance or Moniepoint impact.
