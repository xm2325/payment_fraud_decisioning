# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project built around **temporal modelling, point-in-time SQL/Python features, calibrated risk scores, exact analyst-capacity routing, emerging-fraud detection, delayed labels and monitoring**.

The repository has two evidence layers:

- a transparent **120,000-transaction synthetic payment stream** for controlled stress tests such as unseen attacks, analyst feedback and label delay;
- a reproducible **6,362,620-row PaySim benchmark** for large-scale point-in-time SQL, temporal validation and Fraud Ops routing.

Neither source is Moniepoint production data. No result below is a production impact, prevented-loss or staffing claim.

## v1.8 result snapshot

### Controlled 120k stress tests

- Point-in-time velocity/history increases known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at the stated validation-derived operating point.
- The supervised classifier gets **0% recall** on a future-only shared-device attack; the label-free tail detector gets **93.3% recall at 0.86% legitimate flag rate**.
- At **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**; a governance-fixed 80/20 exploit-explore queue gets **80.9% / 40.2%**.
- A 7-day mature-label view remains at **0% novel recall**, while an invalid instant-label oracle reaches **88.4%**, demonstrating label-latency leakage.
- Verification bias matters: risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels.

### Full 6.36M-row PaySim benchmark

GitHub Actions verifies **6,362,620 transactions, 8,213 fraud cases and steps 1--743**, materialises strict prior-step DuckDB features, and keeps the old/new-balance model as a simulator-mechanics sensitivity only. The locked balance-free reference remains `transaction_plus_relational` with future PR-AUC about **0.350**.

v1.8 adds a three-cycle rolling-origin audit. It compares the frozen v1.7 model/calibrator/routing policy with an as-of refresh that refits on expanding history, recalibrates on a separate recent window and reselects routing alpha only on the immediately preceding policy window.

## v1.7 lifecycle contract: separate calibration from routing selection

v1.7 closed a methodological boundary left explicit in v1.6. The same validation rows no longer both fit probability calibration and choose Fraud Ops routing policy.

The ordered evidence path is:

1. **model training:** steps 1--445;
2. **probability calibration only:** steps 446--519;
3. **routing-policy selection only:** steps 520--594;
4. **untouched future evaluation:** steps 595--743.

The split is determined only from time steps; labels and performance cannot move the cutoff.

| Stage | Fraud rate | Mean predicted probability | Brier | PR-AUC |
|---|---:|---:|---:|---:|
| calibration 446--519 | **1.120%** | 1.125% | 0.00760 | 0.4697 |
| policy selection 520--594 | **0.494%** | **1.662%** | 0.00611 | 0.2673 |
| future test 595--743 | **1.338%** | **2.533%** | 0.01233 | 0.3497 |

The early calibration stage is well matched in mean risk, but the frozen calibrator materially over-predicts in later periods. **Absolute probability calibration is temporally unstable even though rank-based routing remains useful.**

## v1.8 rolling refresh: recalibration is not automatically safer

The rolling schedule keeps the v1.7 stage lengths fixed and moves the test origin forward:

| Cycle | Model training | Calibration | Policy selection | Test |
|---:|---|---|---|---|
| 1 | 1--445 | 446--519 | 520--594 | 595--644 |
| 2 | 1--495 | 496--569 | 570--644 | 645--694 |
| 3 | 1--545 | 546--619 | 620--694 | 695--743 |

Cycle 2 shows why recalibration can help: actual fraud prevalence is **0.818%**, the frozen v1.7 calibrator predicts **2.495%** mean risk, while the refreshed calibrator predicts **1.058%** and lowers Brier loss from **0.00957 to 0.00665**.

Cycle 3 shows the opposite failure mode. Fraud prevalence jumps to **3.863%**; the refreshed calibrator predicts only **1.152%**, so Brier loss worsens from **0.02561 frozen to 0.03015 refreshed**, even though PR-AUC is slightly higher (**0.5272 to 0.5331**). A recent calibrator can still lag a sharp base-rate change.

The default robust `balanced` queue keeps **alpha=0.25 in every cycle**. At 50 reviews/10k, refreshed routing has the same precision and case recall as frozen v1.7 in all three test windows; value recall is identical except for a small **81.74% to 81.54%** decline in cycle 2. This is evidence that calibration and top-k routing must be monitored as different contracts.

A separately governed value-first profile does change. In cycle 3, the preceding policy window selects **alpha=1.0**; on steps 695--743 it raises fraud-value recall from **40.76% to 60.95%** at the same **100% precision and 12.77% case recall**. The preceding cycle slightly worsens, so this is not promoted to a new default.

PaySim has no fraud-label maturity or investigation-completion timestamp. Later rolling cycles may use labels from completed earlier periods, so v1.8 is an **as-of upper-bound under labels being available by the next refresh**, not a production delayed-label result. See `PAYSIM_ROLLING_REFRESH.md` for the complete contract and results.

## Routing policy remains stable under the stricter split

Routing uses the pre-specified family

`priority = P(fraud) × (amount / policy_median_amount)^alpha`

for `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`. Only policy-selection steps 520--594 choose alpha in v1.7, using three contiguous windows and worst-window objectives at exact **50 reviews per 10,000 transactions**.

All declared case-first, balanced and value-first objectives select **alpha=0.25** in the initial lifecycle. v1.8 then tests whether those choices change when the origin moves forward.

| Alpha | Worst policy-window case recall | Worst value recall | Worst balanced H-mean |
|---:|---:|---:|---:|
| 0.00 | 23.64% | 64.78% | 0.3464 |
| **0.25** | **25.39%** | **72.36%** | **0.3822** |
| 0.50 | 25.39% | 72.05% | 0.3776 |
| 0.75 | 25.00% | 72.05% | 0.3775 |
| 1.00 | 24.22% | 72.05% | 0.3678 |

So the alpha=0.25 initial compromise is not an artefact of reusing calibrator-fitting rows for policy tuning.

## Untouched future exact-capacity result

The frozen v1.7 alpha=0.25 routing policy gives across steps 595--743:

| Reviews / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 87.66% | 16.32% | 61.39% |
| 50 | **61.59%** | **22.97%** | **77.67%** |
| 100 | 43.24% | 32.29% | 87.58% |

These figures are almost unchanged from v1.6 despite the stricter lifecycle split. That is evidence of **routing robustness**, not a claim that the probability estimates themselves are stable.

At 50 reviews/10k, frozen future-window fraud-value recall is **75.45% → 81.74% → 40.76%**. In the final high-fraud window every admitted case is fraud, yet case recall is only **12.77%**: analyst capacity, not false-positive ranking, is the limiting factor.

## Exact-capacity routing, not brittle scalar thresholds

Large LightGBM score ties make narrow scalar thresholds unsuitable for exact queue capacity. The project therefore uses an explicit **alerts-per-10,000-transactions** contract. Equal scores are broken only by a stable non-label `event_key`.

The previously documented v1.3 **60.1% precision / 25.9% recall / 80.7% value-recall** narrow-threshold headline is superseded and should not be reused.

## Recipient / mule audit: retained negative evidence

PaySim has no confirmed mule-account label. Standalone prior-step recipient signals remain investigation diagnostics rather than mule classifiers. Recipient fan-in AUC is about **0.493**, the composite recipient-intensity score about **0.467**, and validation-selected recipient thresholds recover **0% future fraud**. The negative result stays visible.

## Architecture

```text
transaction
   -> point-in-time transaction + relational history
   -> balance-free supervised model
   -> calibration-only temporal stage
   -> policy-selection-only temporal stage
   -> exact-capacity alpha-weighted routing
   -> completed prior period
   -> as-of model/calibration/policy refresh
   -> next untouched test period
```

The controlled 120k stream retains a separate anomaly/exploration lane for future-only attacks and remains the source of explicit delayed-label evidence.

## Reproducibility controls

- Equal-timestamp simulator events cannot use one another as history.
- SQL/Python point-in-time parity is executed in tests.
- Full PaySim features use strict prior-step DuckDB windows.
- Stable non-label event keys make loading and exact-capacity tie-breaking deterministic.
- Feature-family selection, calibration, routing-policy selection and future evaluation have explicit temporal boundaries.
- Rolling refresh boundaries are time-only and checked for strict ordering and expanding-history behaviour in unit tests.
- Full PaySim benchmark, monitoring, routing-profile, routing-robustness, stage-separation and rolling-refresh workflows run in GitHub Actions; raw PaySim rows are not committed.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
uvicorn fraud_decisioning.api:app --app-dir src --reload
```

## Key evidence

- `RESULTS.md` — broader controlled and PaySim result narrative.
- `PAYSIM_BENCHMARK.md` — canonical 6.36M-row benchmark contract.
- `PAYSIM_MONITORING.md` — exact-capacity monitoring contract.
- `PAYSIM_ROUTING_PROFILES.md` — validation-selected probability/value routing compromise.
- `PAYSIM_ROUTING_ROBUSTNESS.md` — worst-window routing robustness audit.
- `PAYSIM_STAGE_SEPARATION.md` — v1.7 calibration-versus-policy temporal separation.
- `PAYSIM_ROLLING_REFRESH.md` — v1.8 frozen-versus-refresh rolling-origin audit.
- `APPLICATION_NOTES.md` — safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source and claim boundaries.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

## Data honesty

The 120k simulator deliberately contains a future-only attack that is detectable by historical anomaly signals, so its discovery results are controlled method stress tests. PaySim is also synthetic mobile-money data. Real fraud labels, chargeback maturity, customer outcomes, intervention efficacy, review capacity, device/network quality and production prevalence would all be required before deployment or financial-impact claims.
