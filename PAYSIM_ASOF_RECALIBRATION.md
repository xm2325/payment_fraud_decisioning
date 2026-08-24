# PaySim as-of recalibration under delayed labels — v1.8

v1.8 keeps the v1.7 predictive model, routing alpha (**0.25**) and review capacity (**50 alerts per 10,000 transactions**) fixed. It changes only the probability-calibration governance.

The initial approved calibrator uses steps **446--519**. Later refreshes may add post-519 labels only after they have matured before the next scoring window. A label from transaction step `s` is available before scoring step `t` only when `s + lag < t`.

The maturity lags below are stress-test scenarios, not estimates of Moniepoint label latency.

## Methods

- `frozen_initial`: never refit after the approved steps 446--519 calibrator.
- `asof_24h`: expanding refit using the initial calibration set plus post-519 labels matured by 24 steps.
- `asof_168h`: same contract with a 168-step (7-day) lag.
- `instant_history_diagnostic`: retrospective diagnostic using every prior-step label; not deployable under delayed-label operation.

## Full 6.36M-row result

| Method | Mean 3-window Brier | Worst-window Brier | Mean absolute log risk-ratio error |
|---|---:|---:|---:|
| frozen initial | 0.01578 | **0.02561** | 0.6441 |
| **as-of 24h** | **0.01519** | 0.02849 | **0.3868** |
| as-of 168h | 0.01606 | 0.02647 | 0.7496 |
| instant-history diagnostic | 0.01540 | 0.02905 | 0.5076 |

The 24-hour refresh improves average calibration, but not worst-window calibration. It is therefore not safe to treat “newer labels” as automatic justification for recalibration.

### Future window 1 — steps 595--644

Observed fraud rate: **1.301%**.

- frozen calibrator mean risk: **2.388%**, Brier **0.01215**;
- 24h as-of mean risk: **1.304%**, Brier **0.01043**;
- 7-day refresh has no matured post-519 evidence yet, so it is identical to frozen;
- instant-history diagnostic mean risk: **1.074%**, Brier **0.01043**.

The 24h refresh nearly matches the observed prevalence in this window.

### Future window 2 — steps 645--694

Observed fraud rate: **0.818%**.

- frozen mean risk: **2.495%**, Brier **0.00957**;
- 24h as-of mean risk: **1.080%**, Brier **0.00667**;
- 7-day refresh is still identical to frozen;
- instant-history diagnostic mean risk: **1.160%**, Brier **0.00671**.

The 24h refresh again improves probability quality. At the fixed 50/10k review capacity it also moves fraud-case recall from **25.64% to 26.92%** and fraud-value recall from **81.74% to 82.21%**.

### Future window 3 — steps 695--743

Observed fraud rate jumps to **3.863%**.

- frozen mean risk: **3.133%**, Brier **0.02561**;
- 24h as-of mean risk: **1.603%**, Brier **0.02849**;
- 7-day as-of mean risk: **2.283%**, Brier **0.02647**;
- instant-history diagnostic mean risk: **1.448%**, Brier **0.02905**.

Here the fresher calibrators are worse. Their matured history is dominated by the lower-prevalence period immediately before the abrupt high-fraud window, so they under-predict the new regime.

## Routing stays much more stable than calibration

At fixed `alpha=0.25` and **50 reviews/10k**, recalibration leaves the selected queue unchanged in windows 1 and 3. Only window 2 changes modestly.

This separation is operationally important:

- **absolute probability quality** is sensitive to label maturity and prevalence/concept shift;
- **rank-based exact-capacity routing** is comparatively stable;
- the final high-fraud window remains capacity constrained regardless of recalibration.

## Governance conclusion

v1.8 does **not** recommend automatic recalibration whenever new labels mature. A shorter label delay improved average Brier but worsened the abrupt high-fraud window. A production analogue should use a champion/challenger approval gate with a mature-label holdout and should monitor calibration separately from queue capacity.

The instant-history method is retained only as a diagnostic. The 24h and 168h lags are stress-test assumptions. PaySim is synthetic mobile-money data, so these are methodological benchmark results rather than production impact or a Moniepoint policy recommendation.

Verified full-data GitHub Actions run: **32676300802**.
