# Full PaySim benchmark contract — v1.3

## Dataset audit

The GitHub Actions workflow downloads both public PaySim Parquet shards and aborts unless the combined data match **6,362,620 rows, 8,213 fraud transactions and step range 1--743**. Raw shards are runner-local and are never committed.

## Time contract

PaySim's hourly `step` is split 60% / 20% / 20% into train, validation and future test. All historical features use `RANGE ... AND 1 PRECEDING`, so transactions in the same step cannot use one another as history.

## Feature audit

Four nested model families are evaluated:

1. `transaction_only`: amount and transaction type.
2. `transaction_plus_history`: sender velocity/amount history and recipient fan-in.
3. `transaction_plus_relational`: prior sets plus 7-day sender/recipient activity, sender-recipient pair reuse/amount share and approximate unique counterparties.
4. `full_with_simulator_balances`: relational set plus old/new balance derivatives.

The balance-derived model is sensitivity-only. The final v1.3 implementation chooses the balance-free reference among the first three models using **validation PR-AUC**, then evaluates the frozen choice on future test.

## Verified 6.36M-row relational run

Validation-only balance-free model selection: transaction-only **0.2775**, transaction + history **0.2733**, relational **0.3252** PR-AUC. The relational model is locked before future-test evaluation.

| feature set | PR-AUC | precision | recall | future legit flag | fraud-value recall |
|---|---:|---:|---:|---:|---:|
| transaction only | 0.3403 | 51.0% | 28.2% | 0.367% | 81.16% |
| transaction + history | 0.3408 | 54.4% | 27.6% | 0.314% | 81.15% |
| transaction + relational | **0.3530** | **60.1%** | 25.9% | **0.233%** | 80.66% |
| + simulator balances | 0.9950 | 87.1% | 99.94% | 0.201% | 99.99% |

The relational set improves ranking and precision while reducing legitimate alerts, but it slightly reduces transaction-level and value recall relative to transaction-only at the selected operating point. It is therefore an operational trade-off, not a universal feature win.

A validation-thresholded amount/type rule reaches **46.6% precision, 12.1% recall and 62.2% fraud-value recall** on future test; the selected relational model reaches **60.1%, 25.9% and 80.7%**. Fraud value is highly concentrated (top 25% of fraud cases = **82.4%** of fraud value), so value recall is always shown alongside case recall and the simple-rule baseline.

The very large jump after adding balance derivatives is treated as a simulator-mechanics warning. The portfolio headline stays balance-free.

## Rule and investigation baselines

`isFlaggedFraud` is evaluated as a source rule but is never used as a model feature. On the same future test it produces 8 alerts, 100% precision, 0.48% recall and 1.56% fraud-value recall.

The repo also provides `sql/paysim_rule_backtest.sql`, `sql/paysim_typology_sizing.sql` and `sql/paysim_recipient_investigation.sql`. The latest v1.3 runner additionally computes a validation-thresholded amount/type rule and fraud-value concentration so high value recall can be separated from simple amount concentration.

## Publication contract

The workflow uploads only aggregate audit, split, ablation and operating-point outputs. No raw PaySim rows, feature Parquet or fitted model is published. PaySim is synthetic mobile-money data; results validate the engineering/evaluation path and are not production performance estimates.
