# Payment Fraud Decisioning — Case Study

## Business question

A fraud model is useful only if it changes decisions under operational constraints. This project asks: **which transactions should be approved, challenged, reviewed or blocked when fraud labels are delayed, analyst capacity is limited and new attack patterns may not resemble historical fraud?**

## External benchmark

The external benchmark uses the full public PaySim simulator: **6,362,620 transactions, 8,213 fraud cases, steps 1--743**. The split is strictly chronological: train 1--445, validation 446--594, untouched future test 595--743. Rolling features use only prior steps.

### Model selection before test

Three balance-free candidates are compared using validation PR-AUC only:

| Candidate | Validation PR-AUC |
|---|---:|
| transaction only | 0.2775 |
| transaction + history | 0.2733 |
| transaction + relational history | **0.3252** |

The relational model is selected and frozen before future-test evaluation.

### Future-test result

| Detector | Precision | Fraud recall | Fraud-value recall | Legitimate flag rate |
|---|---:|---:|---:|---:|
| supplied PaySim rule | **100.0%** | 0.48% | 1.56% | 0.000% |
| validation-thresholded amount/type rule | 46.6% | 12.1% | 62.2% | 0.188% |
| selected relational ML model | **60.1%** | **25.9%** | **80.7%** | 0.233% |

The useful result is the increment over an interpretable rule: the ML model more than doubles fraud-case recall while increasing precision and fraud-value coverage.

## Two audits that prevent misleading conclusions

### 1. Simulator balance mechanics

Adding old/new balance derivatives produces validation PR-AUC **0.9923** and future-test PR-AUC **0.9950**. Those fields expose simulator accounting mechanics and are therefore sensitivity-only. They are deliberately excluded from the headline model.

### 2. Fraud-value concentration

PaySim value is concentrated: the largest **10% / 25% / 50%** of fraud cases account for **55.1% / 82.4% / 95.3%** of fraud value. An 80% fraud-value recall is therefore not sufficient evidence by itself; it is reported beside case recall, precision and the amount-based baseline.

## Emerging-fraud path

The repository also contains a controlled unseen-attack simulation. A supervised model trained without the attack gets 0% new-attack recall, while a label-free anomaly channel surfaces the tail pattern. A fixed exploit/explore review queue then trades known-fraud value coverage against discovery, and confirmed analyst outcomes feed a later retraining cycle.

This simulation is a method stress test, not evidence that a particular real fraud typology will be detected at the same rate.

## Fraud Ops layer

The project separates modelling from operations:

- threshold selection on validation rather than test;
- review-capacity frontiers;
- backlog / service-level stress tests;
- delayed-label monitoring;
- verification-bias audits;
- prior-shift calibration checks;
- typology sizing;
- investigation SQL for recipient concentration, pair reuse and rule backtesting;
- explicit reason codes and mitigation playbook.

## Production hand-off

Before deployment I would require event-time device/IP/session signals, account/KYC state, mature fraud labels with label timestamps, intervention outcomes, review-capacity targets and customer-friction outcomes. Feature availability timestamps are part of the contract so the offline benchmark cannot silently use information unavailable at decision time.

## Reproduce

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
```

The full PaySim benchmark is reproducible through `.github/workflows/paysim-full.yml`; raw PaySim rows are downloaded only on the GitHub runner and are not committed.
