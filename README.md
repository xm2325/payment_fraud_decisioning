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

## v1.7 lifecycle contract: separate calibration from routing selection

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

Routing remains stable under the stricter split: all declared case-first, balanced and value-first objectives still choose **alpha=0.25** for

`priority = P(fraud) × (amount / policy_median_amount)^alpha`.

At exact **50 reviews per 10,000 transactions**, the frozen future policy reaches **61.59% precision, 22.97% fraud-case recall and 77.67% fraud-value recall**.

## v1.8: as-of recalibration under delayed labels

v1.8 freezes the predictive model, **alpha=0.25** routing policy and **50 reviews/10k** capacity. It changes only the calibrator-refresh governance.

The initial steps 446--519 calibrator is treated as an already approved deployment artefact. Later refreshes can add post-519 labels only after they have matured before the next future window. The 24h and 168h lags are stress-test scenarios, not estimates of Moniepoint label latency.

| Method | Mean 3-window Brier | Worst Brier | Mean absolute log risk-ratio error |
|---|---:|---:|---:|
| frozen initial | 0.01578 | **0.02561** | 0.6441 |
| **as-of 24h** | **0.01519** | 0.02849 | **0.3868** |
| as-of 168h | 0.01606 | 0.02647 | 0.7496 |
| instant-history diagnostic | 0.01540 | 0.02905 | 0.5076 |

The important result is **not** “recalibrate more often”. In future window 1, 24h as-of refresh moves mean predicted risk from **2.388% to 1.304%** against **1.301% observed fraud prevalence** and improves Brier from **0.01215 to 0.01043**. In window 2 it improves Brier from **0.00957 to 0.00667**.

But in the abrupt high-fraud window 3, observed prevalence rises to **3.863%** while the 24h recalibrator predicts only **1.603%** on average; Brier worsens from **0.02561 to 0.02849**. The instant-history diagnostic is worse still. Matured labels can therefore make a refresh *less* safe when the next regime changes sharply.

At fixed exact review capacity, routing is far more stable than absolute calibration. Windows 1 and 3 select the same queue under all recalibration methods; only window 2 changes modestly, where the 24h refresh raises case recall from **25.64% to 26.92%** and value recall from **81.74% to 82.21%**.

**Governance conclusion:** calibration and queue ranking should be monitored separately. A production analogue should use a mature-label champion/challenger approval gate rather than automatically accepting every newer calibrator.

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
   -> mature-label arrival
   -> as-of calibration challenger
   -> calibration / queue monitoring
```

The controlled 120k stream also retains a separate anomaly/exploration lane for future-only attacks.

## Reproducibility controls

- Equal-timestamp simulator events cannot use one another as history.
- SQL/Python point-in-time parity is executed in tests.
- Full PaySim features use strict prior-step DuckDB windows.
- Stable non-label event keys make loading and exact-capacity tie-breaking deterministic.
- Feature-family selection, calibration, routing-policy selection and future evaluation have explicit temporal boundaries.
- Recalibration refreshes obey explicit label-maturity cutoffs.
- Full PaySim workflows hard-audit the canonical row/fraud/step counts; the v1.8 workflow also caches the verified public Parquet shards to avoid repeated 320MB downloads.

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
- `PAYSIM_ASOF_RECALIBRATION.md` — v1.8 mature-label recalibration audit.
- `APPLICATION_NOTES.md` — safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source and claim boundaries.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

## Data honesty

The 120k simulator deliberately contains a future-only attack that is detectable by historical anomaly signals, so its discovery results are controlled method stress tests. PaySim is also synthetic mobile-money data. Real fraud labels, chargeback maturity, customer outcomes, intervention efficacy, review capacity, device/network quality and production prevalence would all be required before deployment or financial-impact claims.
