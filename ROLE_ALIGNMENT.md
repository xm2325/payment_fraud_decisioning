# Moniepoint Data Scientist (Fraud): project alignment — v0.9

Role page used for alignment: https://moniepoint.com/careers/roles/4737173101

| Role signal | Evidence in this repo |
|---|---|
| Fraud detection models | Logistic baseline + calibrated LightGBM with strict future-period evaluation |
| Feature engineering | Backward-looking sender velocity, amount history, recipient concentration and shared-device/network history |
| Fraud experiments | Review-band randomized-intervention sample-size design; no fabricated treatment effect |
| Customer experience vs fraud loss | Validation-selected approve/review/block policy, full loss/friction frontier and four-scenario policy-assumption audit |
| Fraud typology sizing | Transaction share and value share by simulated attack type |
| Novel fraud vectors | Test-only attack, Isolation Forest, interpretable tail detector and separate anomaly monitoring |
| Fraud Ops capacity | Model-only, hybrid and fixed exploit/explore queues measured at review budgets per 10k, plus hourly 4/6/8-review capacity and traffic-growth stress tests |
| Monitoring and retraining | Rolling-origin backtest, model-score PSI, anomaly early warning, 7-day label-maturity diagnostic, analyst-feedback recovery curve and queue-health monitoring |
| SQL | Point-in-time queries plus executable SQLite/Python parity test including equal-timestamp semantics |
| Python / production | Tested package, model artifact, decision API, CI, Docker and reproducible runners |
| Probability calibration | Base-rate precision translation plus prior-probability correction stress test with an explicit label-shift failure case |
| External validation | Verified full 6.36M-row PaySim GitHub Actions benchmark, balance-field sensitivity audit, relational feature ablation and temporal threshold drift |

The project is a case study, not prior employment in fraud. Synthetic results must remain labelled as such.

## Capacity-aware review control

The project now goes beyond reporting queue overload. It implements a backlog-aware two-lane admission policy that preserves an exploration budget, raises effective review cutoffs as traffic pressure increases, and reports the resulting fraud-value and emerging-fraud coverage loss. This maps the model output to an operational fraud-investigation constraint rather than treating model ranking and Fraud Ops capacity as separate exercises.
