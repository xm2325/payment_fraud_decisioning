# PaySim score-tail surge-capacity audit — v1.6

v1.6 asks a deliberately operational question: **when a fixed review queue becomes capacity-constrained, can an observable model-score signal justify a pre-agreed temporary increase in review capacity?**

The predictive model is unchanged. The routing score is the v1.5 validation-selected compromise:

`priority = calibrated P(fraud) × (amount / validation_amount_scale)^0.25`

The capacity trigger itself uses **calibrated model probability only**, not fraud labels and not future amount outcomes.

## Prospective policy contract

Before future evaluation, the stress-test policy is fixed as follows:

1. On validation, fit the score threshold corresponding to the nominal top 0.5% tail.
2. Record the **actual** validation tail rate because tied scores can make it larger than 0.5%.
3. In each contiguous future window, calculate the share of scores at or above that frozen threshold.
4. Divide by the validation tail rate to obtain a score-tail load multiplier.
5. Reference scenario: if the multiplier is at least **1.5×**, flex review capacity from **50 to 100 alerts per 10,000 transactions**; otherwise remain at 50.
6. Only after the capacity decision, use fraud labels to evaluate precision, fraud-case recall and fraud-value recall.

The top-0.5% tail, 1.5× reference trigger and 50→100 capacity step are scenario assumptions. They are not derived estimates of Moniepoint staffing or an optimal production rule.

## Verified full 6.36M-row result

The canonical PaySim audit again verifies **6,362,620 transactions, 8,213 fraud cases and steps 1–743**. The routing exponent selected on validation remains `alpha=0.25`.

The nominal validation top-0.5% threshold is **0.1547056**. Because the model has tied scores, the realised validation tail rate is **0.6265%**; the future trigger compares against that realised reference rather than pretending the tail is exactly 0.5%.

Future score-tail load:

| future window | step range | score-tail multiplier vs validation | 1.5× reference trigger |
|---|---|---:|:---:|
| 1 | 595–644 | **1.502×** | Yes |
| 2 | 645–694 | **1.661×** | Yes |
| 3 | 695–743 | **2.407×** | Yes |

### Consequence of the reference 50→100/10k flex scenario

| window | added review slots | fixed precision | surge precision | fixed case recall | surge case recall | fixed value recall | surge value recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 212 | 61.32% | 39.86% | 23.55% | **30.62%** | 75.45% | **85.06%** |
| 2 | 334 | 44.44% | 28.64% | 27.11% | **34.98%** | 82.30% | **89.55%** |
| 3 | 72 | 100.00% | **99.30%** | 12.77% | **25.54%** | 40.76% | **81.43%** |

The final window is especially informative. Under the fixed 50/10k queue, all 71 selected cases are fraud, yet case recall is only 12.77%. This is a capacity ceiling rather than a false-positive problem. In the stated flex scenario, 143 cases are reviewed, 142 are fraud, case recall rises to 25.54% and fraud-value recall rises from 40.76% to 81.43%.

That is a **coverage-under-capacity** result, not a prevented-loss estimate. It also does not establish that additional analysts would actually be available on demand.

## Trigger sensitivity — do not optimise on future labels

Window 1 is only barely above the pre-specified 1.5× trigger. v1.6 therefore reports a pre-declared trigger grid `{1.5, 2.0, 2.5}` as governance sensitivity rather than selecting a future-optimal trigger.

Because the measured score-tail multipliers are 1.502×, 1.661× and 2.407×:

| trigger multiplier | windows that would flex | added review slots |
|---:|---|---:|
| **1.5×** | 1, 2, 3 | 618 |
| **2.0×** | 3 only | 72 |
| **2.5×** | none | 0 |

The important finding is sensitivity, not that one row is “best”. A 1.5× governance rule is permissive in this future period; a 2.0× rule would isolate the final high-load window; a 2.5× rule would never flex. Production selection would require business-owned capacity cost, service-level objectives, mature labels and prospective monitoring rather than choosing the most attractive retrospective outcome.

`results/paysim_surge_capacity/surge_trigger_sensitivity.csv` reports the per-window precision/recall/value consequence for every pre-declared trigger. Future labels are used only for consequence measurement.

## Why use score load instead of future fraud prevalence?

Future fraud prevalence is unavailable at decision time because labels mature later. The trigger therefore uses an observable quantity—the mass in the frozen model-score tail. This does not guarantee that score load represents true fraud load; score drift could also be caused by data quality, calibration or transaction-mix changes. A production response would investigate those causes before treating flex capacity as the only action.

## Governance boundary

A practical runbook would separate three questions:

- **Detection:** has the score-tail load moved materially relative to a validation reference?
- **Diagnosis:** is the move caused by fraud mix, feature/data drift, calibration, traffic composition or a system issue?
- **Capacity action:** if the review queue is genuinely saturated and flex capacity is available, which pre-approved capacity tier should be activated?

v1.6 only stress-tests the third question conditional on a simple score-load signal. It does not claim a complete production incident-management policy.

## Evidence boundary

- PaySim is synthetic mobile-money data.
- Future labels never determine whether the surge is triggered.
- The routing exponent was selected on validation before future evaluation.
- The 0.5% score tail, trigger multipliers and 50→100 review capacity are scenario settings.
- More review capacity is not free and is not assumed to exist in production.
- Reported fraud-value recall is captured value among labelled fraud transactions, not prevented monetary loss.
