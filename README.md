# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project built around **temporal modelling, point-in-time SQL/Python features, calibrated risk scores, analyst-capacity routing, emerging-fraud detection, delayed labels and monitoring**.

The repository has two evidence layers:

- a transparent **120,000-transaction synthetic payment stream** for controlled stress tests such as unseen attacks, analyst feedback and label delay;
- a reproducible **6,362,620-row PaySim benchmark** for large-scale point-in-time SQL, model selection and Fraud Ops routing.

Neither source is Moniepoint production data. No result below is a production impact, saved-loss or staffing claim.

## v1.6 result snapshot

### Controlled 120k stress tests

- Point-in-time velocity/history increases known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at the stated validation-derived operating point.
- The supervised classifier gets **0% recall** on a future-only shared-device attack; the label-free tail detector gets **93.3% recall at 0.86% legitimate flag rate**.
- At **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**; a governance-fixed 80/20 exploit-explore queue gets **80.9% / 40.2%**.
- After **100 anomaly-ranked reviews** produce 95 simulated novel confirmations, retrained future novel-fraud recall rises from **0% to 89.3%**.
- A 7-day mature-label view remains at **0% novel recall**, while an invalid instant-label oracle reaches **88.4%**, demonstrating label-latency leakage.
- Verification bias matters: risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels.

### Full 6.36M-row PaySim benchmark

GitHub Actions verifies **6,362,620 transactions, 8,213 fraud cases and steps 1--743**, materialises strict prior-step DuckDB features, then uses train steps 1--445, validation 446--594 and untouched future test 595--743.

Balance-free model selection uses **validation PR-AUC only**:

| Feature family | Validation PR-AUC | Future PR-AUC |
|---|---:|---:|
| transaction only | 0.2790 | 0.3400 |
| + prior-step history | 0.2774 | 0.3406 |
| + relational / pair / counterparty history | **0.3228** | **0.3497** |
| + simulator balance derivatives | 0.9929 | 0.9950 |

The balance-derived row is a simulator-mechanics sensitivity only. The locked balance-free reference is the relational model.

## Fraud Ops contract: exact review capacity

Large LightGBM score ties make a narrow scalar threshold unsuitable for precise analyst-capacity control. v1.4 therefore replaced the old quantile operating point with an explicit **alerts-per-10,000-transactions** contract. Equal model scores are resolved only by a stable non-label `event_key`.

For probability-ranked future queues:

| Reviews / 10k | Precision | Fraud recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | **100.0%** | 7.44% | 32.62% |
| 25 | **96.75%** | 18.02% | 46.39% |
| 50 | **64.34%** | **24.00%** | 71.67% |
| 100 | 41.05% | 30.65% | **83.33%** |

The previously documented v1.3 **60.1% precision / 25.9% recall / 80.7% value-recall** narrow-threshold headline is superseded and should not be reused.

## Same capacity, different business objectives

At the same **50 reviews / 10k**, three rankers expose a real policy trade-off:

| Ranker | Precision | Fraud recall | Fraud-value recall |
|---|---:|---:|---:|
| calibrated fraud probability | **64.34%** | **24.00%** | 71.67% |
| `P(fraud) × amount` | 56.40% | 21.04% | **76.96%** |
| amount/type rule | 42.46% | 15.84% | 70.43% |

Probability ranking is better for clean case capture; expected-loss-style ranking gives up **2.96 percentage points** of case recall to gain **5.28 points** of value recall. `P(fraud) × amount` is a prioritisation heuristic, not prevented-loss estimation.

## v1.5 routing profile: validation-selected compromise

Rather than choose the business objective after looking at future results, v1.5 pre-specifies

`priority = P(fraud) × (amount / validation_median_amount)^alpha`

for `alpha ∈ {0, 0.25, 0.5, 0.75, 1}` and selects alpha using validation labels only. At 50 reviews/10k, **alpha=0.25** wins the declared case-first, balanced and value-first validation objectives.

Frozen on untouched future data, alpha=0.25 gives **61.75% precision, 23.04% fraud-case recall and 77.70% fraud-value recall** at 50 reviews/10k: a deliberate compromise between pure probability and pure expected-loss-style ranking.

## v1.6 robustness check: does alpha=0.25 survive time slicing?

v1.6 divides validation into three contiguous time windows and re-selects routing profiles by **worst-window performance first**, without using future labels. The result is confirmatory: all three robust profiles again select **alpha=0.25**.

At 50 reviews/10k on validation windows:

| Alpha | Worst-window case recall | Worst-window value recall | Worst-window balanced H-mean | Case-recall range |
|---:|---:|---:|---:|---:|
| 0.00 | 25.47% | 69.75% | 0.3731 | 4.18 pp |
| **0.25** | **27.69%** | **77.57%** | **0.4081** | 1.34 pp |
| 0.50 | 26.89% | 77.36% | 0.3992 | 1.01 pp |
| 0.75 | 26.89% | 77.36% | 0.3992 | 1.01 pp |
| 1.00 | 26.89% | 77.36% | 0.3992 | **0.63 pp** |

Alpha=1 is slightly steadier by range alone, but alpha=0.25 has the strongest weakest-window case recall, value recall and balanced objective. Because aggregate and robust selection agree, **v1.6 does not claim a new future-test gain**; it strengthens the validation evidence for the existing policy.

## Capacity saturation is different from model failure

At 50 reviews/10k, the alpha=0.25 future value recall is **75.45% → 82.30% → 40.76%** across three future windows. In the last window all 71 admitted cases are fraud, yet case recall is only **12.77%**. A robust ranker cannot compensate for analyst capacity that is too small for the realised fraud arrival rate.

## Recipient / mule audit: retained negative evidence

PaySim has no confirmed mule-account label. Standalone prior-step recipient signals remain investigation diagnostics rather than mule classifiers. On future test, recipient fan-in AUC is about **0.493**, the composite recipient-intensity score about **0.467**, and validation-selected recipient thresholds recover **0% future fraud**. The negative result stays visible.

## Architecture

```text
transaction
   -> point-in-time transaction + relational history
   -> calibrated supervised risk model ----------> probability queue
   -> alpha-weighted probability × amount --------> governed compromise queue
   -> label-free anomaly channel -----------------> exploration queue
   -> approve / review / block + reason codes
   -> analyst outcome / mature fraud label
   -> as-of retraining + immediate/mature-label monitoring
```

## Reproducibility controls

- Equal-timestamp simulator events cannot use one another as history.
- `sql/sqlite_point_in_time_features.sql` is executed in tests and checked against Python features.
- Full PaySim features use strict prior-step DuckDB windows.
- A stable non-label `event_key` makes full-data loading and capacity tie-breaking deterministic.
- Model, routing alpha and robust-routing selection all use validation-only contracts before future evaluation.
- Full PaySim benchmark, monitoring, routing-profile and routing-robustness workflows are executable in GitHub Actions; raw PaySim rows are not committed.

## Run

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
uvicorn fraud_decisioning.api:app --app-dir src --reload
```

Fast CI-style run:

```bash
FRAUD_N=30000 python scripts/run_all.py
```

## What this repo answers

1. Which future transactions are high risk?
2. How does point-in-time feature engineering change fraud ranking?
3. How should model and routing-policy selection avoid future-test leakage?
4. How many cases and how much fraud value can be covered at a fixed analyst capacity?
5. When should a queue prioritise fraud probability versus expected fraud value?
6. Is a validation-selected routing compromise stable across validation time windows?
7. What happens when fraud arrival rate exceeds analyst capacity?
8. Can label-free signals surface an attack absent from model training?
9. How should review capacity be split between exploitation and emerging-pattern discovery?
10. What changes when fraud labels mature days later?
11. How can investigation-driven labels create verification bias?
12. Which apparently fraud-relevant features fail when tested independently?

## Key evidence

- `RESULTS.md` — complete result narrative and boundaries.
- `PAYSIM_BENCHMARK.md` — 6.36M-row benchmark contract.
- `PAYSIM_MONITORING.md` — exact-capacity and temporal monitoring contract.
- `PAYSIM_ROUTING_PROFILES.md` — validation-selected probability/value routing compromise.
- `PAYSIM_ROUTING_ROBUSTNESS.md` — validation-window worst-case robustness audit.
- `APPLICATION_NOTES.md` — safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source boundaries.
- `MITIGATION_PLAYBOOK.md` — intervention and operations framing.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

Core generated tables from the 120k simulator live under `outputs/tables/`; full PaySim aggregate evidence is produced under `results/`.

## Data honesty

The default simulator contains three known fraud types and one future-only shared-device microburst attack. That attack is intentionally abnormal in historical tail/velocity signals, so anomaly and feedback performance are controlled method stress tests rather than real discovery-rate estimates.

PaySim is synthetic mobile-money data. Real fraud labels, chargeback maturity, customer outcomes, intervention efficacy, review capacity, device/network quality and production prevalence would all be required before deployment or financial-impact claims.
