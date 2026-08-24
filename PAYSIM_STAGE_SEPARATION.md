# PaySim calibration / routing stage separation — v1.7

v1.7 closes a methodological boundary left explicit in v1.6: the same validation period should not both fit the probability calibrator and select the Fraud Ops routing policy.

## Temporal contract

The predictive feature family remains fixed as the balance-free `transaction_plus_relational` LightGBM. This experiment does not re-select the model family.

The canonical full PaySim timeline is now used as four ordered stages:

1. **model training:** steps 1--445;
2. **probability calibration only:** steps 446--519;
3. **routing-policy selection only:** steps 520--594;
4. **untouched future evaluation:** steps 595--743.

The 50/50 validation-stage split is determined only from ordered time steps. Fraud labels and model performance cannot move the cutoff. Future labels never fit the calibrator or select routing alpha.

## Full 6.36M-row result

| Stage | Rows | Fraud cases | Fraud rate | Mean predicted probability | Brier | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| calibration, steps 446--519 | 68,033 | 762 | **1.120%** | 1.125% | 0.00760 | 0.4697 |
| policy selection, steps 520--594 | 160,070 | 790 | **0.494%** | **1.662%** | 0.00611 | 0.2673 |
| future test, steps 595--743 | 123,580 | 1,654 | **1.338%** | **2.533%** | 0.01233 | 0.3497 |

The early calibration stage is well matched in mean predicted risk, but the frozen calibrator materially over-predicts risk in the later policy-selection stage and again over-predicts in the future period. This is evidence that absolute probability calibration is temporally unstable in PaySim; it is not fixed by merely separating the stages.

## Routing selection remains stable

Routing keeps the pre-specified family

`priority = P(fraud) * (amount / policy_median_amount)^alpha`

with `alpha ∈ {0, 0.25, 0.5, 0.75, 1}` and exact 50 reviews per 10,000 transactions during policy selection. The later policy-selection stage is split into three contiguous windows, and alpha is chosen using worst-window objectives only.

All three declared profiles still select **alpha=0.25**:

| Alpha | Worst policy-window case recall | Mean case recall | Worst value recall | Mean value recall | Worst balanced H-mean |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 23.64% | 27.03% | 64.78% | 72.24% | 0.3464 |
| **0.25** | **25.39%** | **27.63%** | **72.36%** | **77.96%** | **0.3822** |
| 0.50 | 25.39% | 27.50% | 72.05% | 77.86% | 0.3776 |
| 0.75 | 25.00% | 27.25% | 72.05% | 77.68% | 0.3775 |
| 1.00 | 24.22% | 26.50% | 72.05% | 77.13% | 0.3678 |

So alpha=0.25 is not an artefact of using the calibrator-fitting rows again for policy tuning.

## Untouched future routing

The frozen alpha=0.25 policy gives:

| Reviews / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 87.66% | 16.32% | 61.39% |
| 50 | **61.59%** | **22.97%** | **77.67%** |
| 100 | 43.24% | 32.29% | 87.58% |

At 50 reviews/10k, future-window value recall remains unstable: **75.45% → 81.74% → 40.76%**. The final high-fraud window remains capacity constrained even though selected routing policy is robust.

## Interpretation

v1.7 strengthens the routing evidence but weakens any claim that a one-time calibrated probability is portable through time. The practical separation is:

- **ranking / capacity routing:** alpha=0.25 survives a stricter temporal selection protocol;
- **absolute risk probability:** the frozen calibrator drifts materially as prevalence and score distributions change;
- **operations:** exact analyst capacity still dominates the final high-fraud window.

A production system would therefore monitor calibration and routing capacity separately. Recalibration would require mature labels and a governance-approved as-of process; future labels cannot be used to retrospectively repair a deployed probability estimate.

PaySim is synthetic mobile-money data. These are methodological benchmark results, not Moniepoint production performance, saved loss or staffing estimates.
