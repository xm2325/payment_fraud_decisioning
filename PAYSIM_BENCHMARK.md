# Full PaySim benchmark contract — v1.4

## Dataset audit

GitHub Actions downloads both public PaySim Parquet shards and aborts unless the combined data match **6,362,620 rows, 8,213 fraud transactions and step range 1--743**. Raw shards are runner-local and are never committed.

## Time contract

PaySim's hourly `step` is split into train steps **1--445**, validation **446--594** and untouched future test **595--743**. All historical features use strict prior-step windows, so transactions in the same hour cannot use one another as history.

A stable non-label `event_key` provides deterministic ordering and tie-breaking. It is never a model feature.

## Feature audit and model selection

Four nested model families are evaluated:

1. `transaction_only`: amount and transaction type;
2. `transaction_plus_history`: sender velocity/amount history and recipient fan-in;
3. `transaction_plus_relational`: prior sets plus sender/recipient activity, sender-recipient pair reuse/amount share and approximate unique counterparties;
4. `full_with_simulator_balances`: relational set plus old/new balance derivatives.

The balance-derived model is sensitivity-only. The balance-free reference model is selected using **validation PR-AUC only**, before future-test evaluation.

Verified v1.4 ranking results:

| Feature set | Validation PR-AUC | Future PR-AUC |
|---|---:|---:|
| transaction only | 0.2790 | 0.3400 |
| transaction + history | 0.2774 | 0.3406 |
| transaction + relational | **0.3228** | **0.3497** |
| + simulator balances | 0.9929 | 0.9950 |

The relational model is therefore the locked balance-free reference. The near-perfect balance-derived result is treated as evidence of PaySim simulator mechanics, not as the portfolio headline.

## Why the old v1.3 threshold numbers are superseded

Earlier v1.3 documentation reported a narrow legitimate-alert operating point using a score quantile. The v1.4 audit found that large LightGBM score ties make that contract unsafe: `score >= threshold` can materially overshoot a narrow budget, while a tie-safe scalar hard cap can materially under-use it.

The old **60.1% precision / 25.9% fraud recall / 80.7% fraud-value recall** relational operating-point headline is therefore **superseded**. Those values should not be used in CV or interview material.

A tie-safe 0.1% validation legitimate-alert hard cap is retained only as a threshold diagnostic. For the relational model it uses only **0.00221%** validation legitimate flags and on future test generates 291 all-fraud alerts, with **17.59% case recall / 44.18% value recall**. This illustrates why scalar thresholds are a poor exact-capacity mechanism when score ties are large.

## Operational evaluation: exact review capacity

The deployable benchmark contract is now total alerts per 10,000 transactions. The top `k` cases are admitted without using labels to determine capacity.

For probability ranking:

| Alerts / 10k | Precision | Fraud-case recall | Fraud-value recall |
|---:|---:|---:|---:|
| 10 | **100.0%** | **7.44%** | 32.62% |
| 25 | **96.75%** | **18.02%** | 46.39% |
| 50 | **64.34%** | **24.00%** | 71.67% |
| 100 | 41.05% | 30.65% | 83.33% |

## Same-capacity routing comparison

Three rankers are compared with exactly the same review slots:

| Alerts / 10k | Ranker | Precision | Fraud recall | Fraud-value recall |
|---:|---|---:|---:|---:|
| 10 | model probability | **100.0%** | **7.44%** | 32.62% |
| 10 | probability × amount | 95.12% | 7.07% | 33.22% |
| 10 | amount/type rule | 74.80% | 5.56% | **33.79%** |
| 25 | model probability | **96.75%** | **18.02%** | 46.39% |
| 25 | probability × amount | 69.16% | 12.88% | **61.45%** |
| 25 | amount/type rule | 53.57% | 9.98% | 54.90% |
| 50 | model probability | **64.34%** | **24.00%** | 71.67% |
| 50 | probability × amount | 56.40% | 21.04% | **76.96%** |
| 50 | amount/type rule | 42.46% | 15.84% | 70.43% |
| 100 | model probability | 41.05% | 30.65% | 83.33% |
| 100 | probability × amount | **42.51%** | **31.74%** | **87.37%** |
| 100 | amount/type rule | 34.41% | 25.70% | 83.03% |

The model contributes most clearly by improving case capture and precision over the simple rule at equal capacity. `P(fraud) × amount` creates a different value-oriented queue: at 50/10k it gives up **2.96 percentage points** of case recall versus probability ranking but gains **5.28 points** of fraud-value recall.

This is a routing-objective trade-off, not a universal model win. `P(fraud) × amount` is a prioritisation heuristic, not an estimate of prevented loss.

## Value concentration

Fraud value is highly concentrated in PaySim: the largest **10% / 25% / 50%** of fraud cases account for about **55.1% / 82.4% / 95.3%** of total fraud value. Value recall is therefore always reported beside case recall, precision and an amount-based baseline.

## Rule and investigation baselines

PaySim's supplied `isFlaggedFraud` is evaluated separately and never used as a feature. On future test it produces **8 alerts**, all fraud, but only **0.48% fraud recall / 1.56% fraud-value recall**.

Standalone recipient-history signals are negative evidence rather than a mule proxy: future AUC is around 0.47--0.49 and validation-selected thresholds recover **0% future fraud**. PaySim does not provide a confirmed mule-account label.

## Publication contract

The full benchmark and monitoring workflows upload only aggregate audit, split, ranking and capacity outputs. No raw PaySim rows, feature Parquet or fitted model is published. PaySim is synthetic mobile-money data; these results validate the engineering/evaluation path and are not production performance estimates.
