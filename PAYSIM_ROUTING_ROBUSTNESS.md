# PaySim routing robustness — v1.6

v1.6 asks whether the v1.5 routing choice is robust **inside the validation period**, rather than merely optimal after aggregating all validation rows.

## Selection contract

The predictive model is unchanged: the balance-free `transaction_plus_relational` LightGBM is trained on steps 1--445, calibrated on the existing validation period, and evaluated on untouched future steps 595--743.

Routing uses the pre-specified alpha family

`priority = P(fraud) * (amount / validation_median_amount)^alpha`

with `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`. Alpha=0 is pure probability ranking and alpha=1 is expected-loss-style `P(fraud) × amount` ranking. Intermediate values are routing-policy compromises, not new model features.

For robustness selection:

1. fit the amount scale once on the full validation period;
2. split validation steps 446--594 into three contiguous windows;
3. evaluate every alpha at the same exact **50 reviews per 10,000 transactions** in every validation window;
4. select case-first, balanced and value-first policies by their **worst validation window first**, then mean validation performance;
5. freeze the selected alpha before any future-test evaluation.

Future labels never choose alpha.

## Full 6.36M-row result

The aggregate v1.5 selector and the new worst-window selector independently choose **alpha=0.25** for all three routing profiles.

| Alpha | Worst-window case recall | Mean case recall | Worst-window value recall | Mean value recall | Worst-window balanced H-mean | Case-recall range |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 25.47% | 28.00% | 69.75% | 73.88% | 0.3731 | 4.18 pp |
| **0.25** | **27.69%** | **28.21%** | **77.57%** | **79.04%** | **0.4081** | 1.34 pp |
| 0.50 | 26.89% | 27.38% | 77.36% | 78.68% | 0.3992 | 1.01 pp |
| 0.75 | 26.89% | 27.38% | 77.36% | 78.68% | 0.3992 | 1.01 pp |
| 1.00 | 26.89% | 27.25% | 77.36% | 78.61% | 0.3992 | **0.63 pp** |

Alpha=1 has the smallest case-recall range, but its weakest-window recall and balanced performance are lower. Alpha=0.25 therefore wins the declared robust objectives without requiring future-test tuning.

Relative to probability-only routing, alpha=0.25 raises the weakest validation-window case recall by **2.22 percentage points** and fraud-value recall by **7.82 points**, while reducing the case-recall range from **4.18 to 1.34 points**.

## Untouched future evaluation

Because aggregate and robust validation selection agree on alpha=0.25, v1.6 does **not** claim a new future-test gain. The frozen future result is intentionally identical to v1.5.

| Review capacity / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 88.64% | 16.51% | 60.99% |
| 50 | **61.75%** | **23.04%** | **77.70%** |
| 100 | 43.24% | 32.29% | 87.58% |

At 50 reviews/10k, the three future windows still show substantial operational instability:

- steps 595--644: precision 61.32%, case recall 23.55%, value recall 75.45%;
- steps 645--694: precision 44.44%, case recall 27.11%, value recall 82.30%;
- steps 695--743: precision 100%, case recall 12.77%, value recall 40.76%.

The last window remains a capacity-saturation problem. A temporally robust routing alpha cannot compensate for a review budget that is too small for the realised fraud arrival rate.

## Interpretation

The v1.6 result is **robustness confirmation**, not another round of future-test optimisation. It strengthens the v1.5 alpha=0.25 policy because the same choice survives a stricter validation-only worst-window criterion.

The next methodological boundary is that the existing validation period is still used both to fit the probability calibrator and to select routing policy. A stronger next experiment would pre-specify separate calibration and policy-selection subperiods before the untouched future test.

PaySim is synthetic mobile-money data. These results validate temporal evaluation and Fraud Ops routing methodology; they are not production performance, prevented-loss or Moniepoint impact estimates.
