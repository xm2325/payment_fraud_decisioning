# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project built around **temporal modelling, point-in-time SQL/Python features, calibrated risk scores, exact analyst-capacity routing, emerging-fraud detection, delayed labels and monitoring**.

The repository has two evidence layers:

- a transparent **120,000-transaction synthetic payment stream** for controlled stress tests such as unseen attacks, analyst feedback and label delay;
- a reproducible **6,362,620-row PaySim benchmark** for large-scale point-in-time SQL, temporal validation and Fraud Ops routing.

Neither source is Moniepoint production data. No result below is a production impact, prevented-loss or staffing claim.

## v1.7 result snapshot

### Controlled 120k stress tests

- Point-in-time velocity/history increases known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at the stated validation-derived operating point.
- The supervised classifier gets **0% recall** on a future-only shared-device attack; the label-free tail detector gets **93.3% recall at 0.86% legitimate flag rate**.
- At **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**; a governance-fixed 80/20 exploit-explore queue gets **80.9% / 40.2%**.
- A 7-day mature-label view remains at **0% novel recall**, while an invalid instant-label oracle reaches **88.4%**, demonstrating label-latency leakage.
- Verification bias matters: risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels.

### Full 6.36M-row PaySim benchmark

GitHub Actions verifies **6,362,620 transactions, 8,213 fraud cases and steps 1--743**, materialises strict prior-step DuckDB features, and keeps the old/new-balance model as a simulator-mechanics sensitivity only. The locked balance-free reference remains `transaction_plus_relational` with future PR-AUC about **0.350**.

## v1.7 lifecycle contract: separate calibration from routing selection

v1.7 closes a methodological boundary left explicit in v1.6. The same validation rows no longer both fit probability calibration and choose Fraud Ops routing policy.

The ordered evidence path is now:

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

## Routing policy remains stable under the stricter split

Routing uses the pre-specified family

`priority = P(fraud) × (amount / policy_median_amount)^alpha`

for `alpha ∈ {0, 0.25, 0.5, 0.75, 1}`. Only policy-selection steps 520--594 choose alpha, using three contiguous windows and worst-window objectives at exact **50 reviews per 10,000 transactions**.

All declared case-first, balanced and value-first objectives still choose **alpha=0.25**.

| Alpha | Worst policy-window case recall | Worst value recall | Worst balanced H-mean |
|---:|---:|---:|---:|
| 0.00 | 23.64% | 64.78% | 0.3464 |
| **0.25** | **25.39%** | **72.36%** | **0.3822** |
| 0.50 | 25.39% | 72.05% | 0.3776 |
| 0.75 | 25.00% | 72.05% | 0.3775 |
| 1.00 | 24.22% | 72.05% | 0.3678 |

So the alpha=0.25 compromise is not an artefact of reusing calibrator-fitting rows for policy tuning.

## Untouched future exact-capacity result

The frozen alpha=0.25 routing policy gives:

| Reviews / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | 100.0% | 7.44% | 33.93% |
| 25 | 87.66% | 16.32% | 61.39% |
| 50 | **61.59%** | **22.97%** | **77.67%** |
| 100 | 43.24% | 32.29% | 87.58% |

These figures are almost unchanged from v1.6 despite the stricter lifecycle split. That is evidence of **routing robustness**, not a claim that the probability estimates themselves are stable.

At 50 reviews/10k, future-window fraud-value recall remains **75.45% → 81.74% → 40.76%**. In the final high-fraud window every admitted case is fraud, yet case recall is only **12.77%**: analyst capacity, not false-positive ranking, is the limiting factor.

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
   -> analyst outcome / mature fraud label
   -> as-of retraining + calibration / capacity monitoring
```

The controlled 120k stream also retains a separate anomaly/exploration lane for future-only attacks.

## Reproducibility controls

- Equal-timestamp simulator events cannot use one another as history.
- SQL/Python point-in-time parity is executed in tests.
- Full PaySim features use strict prior-step DuckDB windows.
- Stable non-label event keys make loading and exact-capacity tie-breaking deterministic.
- Feature-family selection, calibration, routing-policy selection and future evaluation have explicit temporal boundaries.
- Full PaySim benchmark, monitoring, routing-profile, routing-robustness and stage-separation workflows run in GitHub Actions; raw PaySim rows are not committed.

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
- `APPLICATION_NOTES.md` — safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source and claim boundaries.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

## Data honesty

The 120k simulator deliberately contains a future-only attack that is detectable by historical anomaly signals, so its discovery results are controlled method stress tests. PaySim is also synthetic mobile-money data. Real fraud labels, chargeback maturity, customer outcomes, intervention efficacy, review capacity, device/network quality and production prevalence would all be required before deployment or financial-impact claims.
