# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project built around **temporal modelling, point-in-time SQL/Python features, calibrated risk scores, analyst-capacity routing, emerging-fraud detection, delayed labels and monitoring**.

The repository has two evidence layers:

- a transparent **120,000-transaction synthetic payment stream** used for controlled stress tests such as unseen attacks, analyst feedback and label delay;
- a reproducible **6,362,620-row PaySim benchmark** used for large-scale point-in-time SQL, model selection and Fraud Ops capacity routing.

Neither source is Moniepoint production data, and no result below is a production impact or saved-loss claim.

## v1.4 result snapshot

### Controlled 120k stress tests

- Point-in-time velocity/history increases known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at the stated validation-derived operating point.
- The supervised classifier gets **0% recall** on a future-only shared-device attack; the label-free tail detector gets **93.3% recall at 0.86% legitimate flag rate**.
- At **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**; a governance-fixed 80/20 exploit-explore queue gets **80.9% / 40.2%**.
- After **100 anomaly-ranked reviews** produce 95 simulated novel confirmations, retrained future novel-fraud recall rises from **0% to 89.3%**.
- A 7-day mature-label view remains at **0% novel recall**, while an invalid instant-label oracle reaches **88.4%**, demonstrating label-latency leakage.
- Verification bias matters: risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels.
- Fraud Ops capacity is stress-tested separately from model quality: a fixed 6-review/hour scenario is feasible at 1x traffic but becomes overloaded as traffic rises. These staffing values are scenario assumptions, not recommendations.

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

v1.4 replaced a brittle scalar-threshold operating point with an explicit **alerts-per-10,000-transactions** capacity contract. Large LightGBM score ties made the old narrow quantile threshold unsafe; the previously documented v1.3 **60.1% precision / 25.9% recall / 80.7% fraud-value recall** operating-point headline is superseded and should not be reused.

For probability-ranked review queues on the future PaySim period:

| Reviews / 10k | Precision | Fraud recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | **100.0%** | **7.44%** | 32.62% |
| 25 | **96.75%** | **18.02%** | 46.39% |
| 50 | **64.34%** | **24.00%** | 71.67% |
| 100 | 41.05% | 30.65% | 83.33% |

Equal model scores are resolved only by a stable non-label `event_key`; labels are never used to define queue capacity.

## Same capacity, different business objectives

The strongest v1.4 operational result is not a single winning score. The project compares three rankers using **exactly the same review slots**:

1. calibrated fraud probability — prioritises likely fraud cases;
2. `P(fraud) × amount` — a simple expected-loss prioritisation heuristic;
3. `TRANSFER/CASH_OUT + amount` — an interpretable baseline.

At **50 reviews / 10k**:

| Ranker | Precision | Fraud recall | Fraud-value recall |
|---|---:|---:|---:|
| model probability | **64.34%** | **24.00%** | 71.67% |
| probability × amount | 56.40% | 21.04% | **76.96%** |
| amount/type rule | 42.46% | 15.84% | 70.43% |

Probability ranking is better for clean case capture; expected-loss ranking gives up **2.96 percentage points** of case recall to gain **5.28 points** of value recall. At 100/10k, expected-loss ranking reaches **42.51% precision / 31.74% case recall / 87.37% value recall**, slightly ahead of probability ranking on all three metrics in that future period.

`P(fraud) × amount` is a queue-prioritisation heuristic, not prevented-loss estimation.

## Capacity saturation is different from model failure

At a fixed **50 reviews / 10k**, probability-ranked fraud-value recall across three future windows is **74.34% → 80.99% → 40.76%**. In the last window fraud prevalence rises to **3.86%**. All 71 admitted cases are fraud, yet fraud-case recall is only **12.77%**.

That is a queue-capacity problem rather than a false-positive problem. In the same high-fraud window, value recall is **40.76%** for probability ranking, **61.97%** for probability × amount and **64.78%** for the amount/type rule, showing why ranking objective and capacity monitoring must be separated.

## Recipient / mule audit: retained negative evidence

PaySim has no confirmed mule-account label. Standalone prior-step recipient signals therefore remain investigation diagnostics rather than mule classifiers. On future test, recipient fan-in AUC is about **0.493**, the composite recipient-intensity score about **0.467**, and validation-selected recipient thresholds recover **0% future fraud**.

The negative result stays in the repository rather than being hidden.

## Architecture

```text
transaction
   -> point-in-time transaction + history features
   -> calibrated supervised risk model ----------> probability queue
   -> risk × amount ------------------------------> value-priority queue
   -> label-free anomaly channel -----------------> exploration queue
   -> approve / review / block + reason codes
   -> analyst outcome / mature fraud label
   -> as-of training set + feedback retraining
   -> immediate + matured-label monitoring
```

## Point-in-time and reproducibility controls

- Python processes equal-timestamp simulator events as a batch, so same-time events cannot see one another.
- `sql/sqlite_point_in_time_features.sql` is executed in tests and checked against Python features.
- Full PaySim features use strict prior-step DuckDB windows.
- A stable non-label `event_key` makes full-data split loading and capacity tie-breaking deterministic.
- Independent full-benchmark and monitoring workflows reproduce the relational model outputs to numerical precision.

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

Full PaySim benchmark and monitoring are run in GitHub Actions; raw PaySim rows are not committed. Aggregate outputs are written under `results/paysim_full/` and `results/paysim_monitoring/` on main-branch runs.

## What this repo answers

1. Which future transactions are high risk?
2. How does point-in-time feature engineering change fraud ranking?
3. How should model selection avoid future-test leakage?
4. How many cases and how much fraud value can be covered at a fixed analyst capacity?
5. When should a queue prioritise fraud probability versus expected fraud value?
6. What happens when fraud arrival rate exceeds analyst capacity?
7. Can label-free signals surface an attack absent from model training?
8. How should a fixed review budget be split between exploitation and emerging-pattern discovery?
9. What changes when fraud labels mature days later?
10. Can anomaly investigations produce useful retraining labels?
11. How can investigation-driven labels create verification bias?
12. How do fraud prevalence and calibration assumptions change precision planning?
13. Which apparently fraud-relevant features fail when tested independently?

## Key evidence

- `RESULTS.md` — complete result narrative and boundaries.
- `PAYSIM_BENCHMARK.md` — 6.36M-row external benchmark contract.
- `PAYSIM_MONITORING.md` — scalar-threshold diagnostic, exact-capacity routing and future-window monitoring.
- `APPLICATION_NOTES.md` — safe CV/interview wording.
- `DATA_PROVENANCE.md` — data-source boundaries.
- `MITIGATION_PLAYBOOK.md` — intervention and operations framing.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study preparation.

Core generated tables from the 120k simulator live under `outputs/tables/`; full PaySim aggregate evidence is produced under `results/`.

## Data honesty

The default simulator contains three known fraud types and one future-only shared-device microburst attack. That attack is intentionally abnormal in historical tail/velocity signals, so anomaly and feedback performance are controlled method stress tests rather than real discovery-rate estimates.

PaySim is synthetic mobile-money data. Real fraud labels, chargeback maturity, customer outcomes, intervention efficacy, review capacity, device/network quality and production prevalence would all be required before making deployment or financial-impact claims.
