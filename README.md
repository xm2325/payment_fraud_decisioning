# Payment Fraud Decisioning & Early-Warning Workbench

A Moniepoint-aligned fraud data-science portfolio project: temporal risk modelling, point-in-time Python/SQL features, loss/friction policy design, fraud typology sizing, unseen-attack detection, analyst-capacity routing, delayed labels, analyst feedback and monitoring.

## v1.3 result snapshot

The first result snapshot below is from the repository's transparent **120,000-transaction synthetic payment stream**. A separate section reports a verified full 6.36M-row PaySim benchmark. Neither source is Moniepoint data or a production impact claim.

- Point-in-time velocity/history raises known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%** at a validation-derived operating point with about **1.15%** legitimate flags.
- Adding network-style history to the supervised model does not improve known-fraud PR-AUC (**0.640 -> 0.635**), so those signals stay in the anomaly/investigation path.
- The supervised model gets **0% recall** on a test-only shared-device attack. The label-free tail detector gets **93.3% recall at 0.86% legitimate flag rate**.
- At **200 reviews per 10,000 transactions**, model-only routing gets **81.8% fraud-value recall and 0% new-attack recall**; a fixed 80/20 exploit-explore queue gets **80.9% value recall and 40.2% new-attack recall**.
- Analyst feedback closes the loop: with no emerging-attack labels, future novel recall is **0%**; after the top **100 anomaly-ranked reviews** from the discovery window yield **95 confirmed novel cases**, retrained future novel recall reaches **89.3%**.
- A 7-day matured-label view gets **0% new-attack recall** on days 54-59, while an invalid instant-label oracle gets **88.4%**. This directly demonstrates label-latency leakage.
- Investigation-driven verification can bias the training set. In a synthetic follow-up stress test, risk-triggered labels alone give known-fraud PR-AUC **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full historical labels. A 10% random audit outside the triggered set raises mean PR-AUC to **0.618**.
- Base rate matters: keeping the measured detector TPR/FPR fixed, expected supervised-alert precision is only **3.0%** at **0.1% fraud prevalence**, versus **23.8%** at 1%. Synthetic observed precision must not be transferred directly to production planning.
- Policy impact is assumption-sensitive: across four pre-specified efficacy/cost scenarios, the validation-selected **block threshold stays at 0.25**, review threshold ranges **0.03-0.06**, future fraud-value prevention ranges **65.6-84.3%**, and legitimate friction ranges **2.72-4.88%**.
- Probability calibration is also base-rate dependent. At an emulated **0.10% fraud prevalence**, leaving the validation-calibrated posterior unchanged gives mean predicted risk **0.679%** and Brier **0.001665**; prior correction gives **0.085%** and **0.000855**. At 2% prevalence the correction slightly worsens Brier, showing that pure label shift does not explain attack-driven concept drift.
- Fraud Ops capacity is now stress-tested in time, not only as a fixed top-k budget. The reference review + exploration policy produces about **5.42 candidates/hour**. At **4 reviews/hour**, 1x traffic ends the 12-day window with **409 queued cases**; **6 reviews/hour** meets the four-hour wait proxy at 1x traffic, while **8 reviews/hour** fails it at 1.5x traffic. Staffing numbers and the four-hour target are scenario assumptions.
- v1.0 closed the queue-capacity loop with backlog-aware admission control. At a fixed **6 reviews/hour**, candidate acceptance is **88.3%** at 1x traffic and **67.8%** at 1.5x. Automatic blocks plus capacity-admitted reviews cover **94.0%** of fraud value at 1x and **91.3%** at 1.5x; novel-fraud recall changes from **81.2%** to **60.3%**. At 4x traffic, protecting capacity requires accepting only **27.2%** of candidates and novel recall falls to **20.5%**, exposing the discovery cost of SLA protection rather than hiding it in backlog.

- Full PaySim audit: on 6.36M rows, the **validation-only selected** balance-free champion is the relational model (validation PR-AUC **0.3252** vs **0.2775** transaction-only). On the untouched future test it reaches PR-AUC **0.3530**, precision **60.1%**, fraud recall **25.9%**, and fraud-value recall **80.7%**. A validation-thresholded amount/type rule reaches only **46.6% / 12.1% / 62.2%** respectively, showing useful incremental value beyond a simple high-amount rule.
See `RESULTS.md` for complete tables and boundaries.

## What this repo answers

1. Which future transactions are high risk?
2. How much fraud value can be stopped for a given customer-friction level?
3. Which fraud types account for the largest loss?
4. Can label-free signals surface an attack absent from model training?
5. How should a fixed analyst budget be split between known-risk cases and emerging-pattern discovery?
6. How does performance change across rolling future windows?
7. What happens when confirmed fraud labels arrive days later?
8. Can anomaly investigations produce useful new labels for retraining?
9. How can investigation-driven labels leave some fraud typologies under-covered?
10. How does alert precision change when deployment fraud prevalence differs from the synthetic benchmark?
11. How should calibrated probabilities be corrected when only the fraud prior changes, and when does that assumption fail?
12. At what traffic and staffing levels does the Fraud Ops queue breach a simple wait-time proxy?
13. When a queue is overloaded, how much fraud coverage and emerging-fraud discovery are lost by capacity-aware admission control?

## Architecture

```text
transaction
   -> point-in-time transaction + history features
   -> calibrated supervised risk model ---------> exploit queue
   -> label-free anomaly channel ---------------> exploration queue
   -> approve / review / block + reason codes
   -> analyst outcome / mature fraud label
   -> as-of training set + feedback retraining
   -> immediate + matured-label monitoring
```

The supervised and anomaly channels have different jobs. The supervised model targets known fraud with mature labels. The anomaly channel is an early-warning and investigation signal; analyst-confirmed findings can later become supervised training evidence.

## Point-in-time SQL parity

The Python feature builder now processes equal-timestamp events as a batch: events with the same recorded timestamp cannot use one another as history. `tests/test_sql_parity.py` executes the SQLite reference query in `sql/sqlite_point_in_time_features.sql` and verifies equality with Python for sender, recipient and device rolling counts, including a timestamp-tie fixture.

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

The current test suite has **27 tests**, including point-in-time feature tests, SQL/Python parity, PaySim schema adaptation, monitoring, experiment sizing, API behaviour, analyst-feedback execution and verification-bias execution.

## PaySim external benchmark — verified full 6.36M run

GitHub Actions runs the complete public PaySim table from two Parquet shards and hard-fails unless it sees **6,362,620 transactions, 8,213 fraud cases and steps 1--743**. DuckDB computes strict prior-step features before a 60/20/20 time split: train steps 1--445, validation 446--594 and untouched future test 595--743.

Model choice is made **only on validation PR-AUC** among balance-free candidates. The relational model wins validation (**0.3252**) over transaction-only (**0.2775**) and transaction + history (**0.2733**), so it is locked before future-test evaluation.

| PaySim feature set | Validation PR-AUC | Test PR-AUC | Fraud recall | Precision | Test legitimate flag rate | Fraud-value recall |
|---|---:|---:|---:|---:|---:|---:|
| transaction only | 0.2775 | 0.3403 | **28.2%** | 51.0% | 0.367% | **81.16%** |
| + prior-step sender/recipient history | 0.2733 | 0.3408 | 27.6% | 54.4% | 0.314% | 81.15% |
| + relational / pair / counterparty history | **0.3252** | **0.3530** | 25.9% | **60.1%** | **0.233%** | 80.66% |
| + simulator balance derivatives | 0.9923 | 0.9950 | 99.94% | 87.1% | 0.201% | 99.99% |

The relational champion improves ranking and operational selectivity, but not every coverage metric. Relative to transaction-only it produces fewer alerts and fewer legitimate flags, with higher precision, while sacrificing some transaction recall. That trade-off is kept visible rather than summarised as a universal gain.

A deliberately simple **TRANSFER/CASH_OUT + amount** rule is also thresholded on validation before future testing. It reaches **46.6% precision, 12.1% fraud recall and 62.2% fraud-value recall**. The validation-selected relational model reaches **60.1%, 25.9% and 80.7%** on the same future period. This is the main external-benchmark increment: the ML model adds meaningful case coverage and value coverage beyond an interpretable amount/type rule.

Fraud value is highly concentrated in PaySim: the largest **10% of fraud cases account for 55.1%** of fraud value, the largest **25% account for 82.4%**, and the largest **50% account for 95.3%**. For that reason, fraud-value recall is always reported beside transaction recall, precision and a simple amount-based baseline; an 80% value-recall number alone would overstate model contribution.

The old/new-balance-derived model remains a sensitivity check only. Its near-perfect PR-AUC is evidence that PaySim simulator accounting fields can make fraud unusually easy to separate; it is not the portfolio headline and must not be translated into a production claim. PaySim's supplied `isFlaggedFraud` rule is evaluated separately and never used as a feature; on the same future test it generates 8 alerts, 100% precision, **0.48% fraud recall** and **1.56% fraud-value recall**.

The final v1.3 full-data workflow completed successfully on GitHub Actions in about **216 seconds** after download. Raw transactions, materialised features and fitted models remain runner-local; only aggregate files in `results/paysim_full/` are retained. PaySim is synthetic mobile-money data, so these are external engineering/evaluation results rather than production fraud estimates.

The repo also includes SQL investigations and rule backtests so the external benchmark can be discussed as a Fraud Ops / investigation problem, not only as a model benchmark. See `PAYSIM_BENCHMARK.md`, `DATA_PROVENANCE.md`, `MITIGATION_PLAYBOOK.md` and `TAKE_HOME_WALKTHROUGH.md`.

## Key outputs

```text
outputs/tables/feature_ablation.csv
outputs/tables/policy_frontier_test.csv
outputs/tables/fraud_typology_sizing.csv
outputs/tables/novelty_detection_metrics.csv
outputs/tables/review_capacity_metrics.csv
outputs/tables/exploration_share_sensitivity.csv
outputs/tables/rolling_temporal_backtest.csv
outputs/tables/delayed_label_retraining.csv
outputs/tables/analyst_feedback_curve.csv
outputs/tables/verification_bias_sensitivity.csv
outputs/tables/fraud_prevalence_precision_sensitivity.csv
outputs/tables/policy_assumption_sensitivity.csv
outputs/tables/prior_shift_calibration.csv
outputs/tables/queue_sla_stress.csv
outputs/tables/adaptive_capacity_routing.csv
outputs/tables/sample_investigation_queue.csv
outputs/tables/weekly_monitoring.csv
```

A concise recruiter-facing narrative is in `CASE_STUDY.md`. Running `python scripts/build_report.py` also writes a self-contained HTML case study to `outputs/moniepoint_fraud_case_study.html`.

## Data honesty

The default simulator contains three known fraud types and one shared-device microburst attack that starts only in the future test period. The unseen attack is intentionally detectable by historical tail/velocity signals, so its anomaly and feedback results are a controlled method stress test rather than a claim about real emerging fraud. Real fraud labels, chargeback maturity, intervention outcomes, review capacity and device/network quality would be needed before making production claims.
