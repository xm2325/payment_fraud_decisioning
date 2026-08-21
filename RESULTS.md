# Result snapshot — v1.4

## 1. Full PaySim external benchmark

A GitHub-hosted run verifies the canonical **6,362,620-row / 8,213-fraud / step-1-to-743** PaySim table and evaluates a strict future split: train steps 1--445, validation 446--594 and future test 595--743. Fraud prevalence rises materially across the chronology, reaching **1.3384%** in the future test.

Balance-free model selection uses validation PR-AUC only:

| Feature family | Validation PR-AUC | Future PR-AUC |
|---|---:|---:|
| transaction only | 0.2790 | 0.3400 |
| + prior-step history | 0.2774 | 0.3406 |
| + relational history | **0.3228** | **0.3497** |
| + simulator balances | 0.9929 | 0.9950 |

The relational model is locked before future-test evaluation. The simulator-balance row is retained only as a dataset-mechanics sensitivity.

### v1.4 threshold correction

The old v1.3 quantile-threshold headline (**60.1% precision / 25.9% recall / 80.7% fraud-value recall**) is superseded. Large score ties meant the old threshold rule did not enforce the stated narrow alert budget.

The corrected scalar threshold is a **diagnostic only**. A tie-safe target of 0.1% validation legitimate flags uses only **0.00221%** actual validation legitimate flags; on future test it generates 291 all-fraud alerts with **17.59% case recall / 44.18% value recall**. The under-use itself is evidence that a scalar threshold is not a reliable exact-capacity mechanism here.

### Exact-capacity routing

The operational benchmark therefore fixes total analyst alerts per 10,000 transactions and ranks cases without using labels to set capacity.

| Alerts / 10k | Probability precision | Probability recall | Probability value recall |
|---:|---:|---:|---:|
| 10 | **100.0%** | **7.44%** | 32.62% |
| 25 | **96.75%** | **18.02%** | 46.39% |
| 50 | **64.34%** | **24.00%** | 71.67% |
| 100 | 41.05% | 30.65% | 83.33% |

### Same-capacity routing objectives

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

At **50 alerts / 10k**, probability ranking improves case recall by **8.16 percentage points** and precision by **21.88 points** versus the amount/type baseline. `P(fraud) × amount` gives up **2.96 points** of case recall versus probability ranking but gains **5.28 points** of fraud-value recall.

The expected-loss score is a prioritisation heuristic, not prevented loss.

### Future-window capacity saturation

At 50 alerts / 10k, probability-ranked fraud-value recall is **74.34% → 80.99% → 40.76%** across the three future windows. In the final window fraud prevalence is **3.86%**; all 71 reviewed cases are fraud, yet case recall is only **12.77%**. The failure is insufficient analyst capacity relative to fraud arrivals, not excessive false positives.

In that same window all three rankers have 100% precision / 12.77% case recall, but value recall is **40.76% probability**, **61.97% probability × amount**, and **64.78% amount/type rule**.

Fraud value is concentrated: the largest **10% / 25% / 50%** of fraud cases account for about **55.1% / 82.4% / 95.3%** of fraud value. Value recall is therefore never interpreted alone.

## 2. Supervised feature ablation on the 120k controlled simulator

Known-fraud evaluation at the stated validation-derived operating point:

| Feature set | PR-AUC | Fraud recall | Fraud-value recall |
|---|---:|---:|---:|
| Static transaction features | 0.597 | 60.2% | 78.1% |
| + point-in-time velocity/history | **0.640** | **66.9%** | **86.1%** |
| + network-style history | 0.635 | 64.3% | 84.1% |

Network-style history does not improve the supervised champion in this run, so those signals remain available for anomaly/investigation use rather than being forced into the classifier.

## 3. Decision-policy sensitivity

Under the stated simulator intervention assumptions, the reference validation-selected review/block policy covers **78.5%** of future fraud value with **4.88%** legitimate friction. Across four pre-specified cost/efficacy scenarios, future fraud-value prevention ranges **65.6--84.3%** and legitimate friction **2.72--4.88%**.

These are simulated policy scenarios, not estimates of real Moniepoint loss prevention or customer cost.

## 4. Unseen-attack stress test

| Detector | Known-fraud recall | New-attack recall | Legitimate flag rate |
|---|---:|---:|---:|
| Supervised | 66.9% | **0.0%** | 1.15% |
| Isolation Forest | 2.6% | 31.8% | 0.90% |
| Interpretable tail detector | 0.4% | **93.3%** | 0.86% |
| Supervised OR tail | 66.9% | **93.3%** | 2.00% |

The future-only attack was deliberately designed to be abnormal in shared-device/velocity history while looking ordinary to common supervised cues. This is a controlled method stress test.

## 5. Exploit/explore review capacity

At **200 reviews / 10k**, model-only routing gets **81.8% fraud-value recall / 0% new-attack recall**. A fixed 80/20 exploit-explore split gets **80.9% value recall / 40.2% new-attack recall**.

The exploration share is governance-defined, not tuned on future novel-fraud labels.

## 6. Analyst feedback and label maturity

The anomaly channel ranks a discovery window for review, confirmed cases are added to training, and evaluation occurs on a later window.

| Reviews | Confirmed novel cases | Future novel recall | Future fraud-value recall |
|---:|---:|---:|---:|
| 0 | 0 | 0.0% | 80.0% |
| 10 | 10 | 38.0% | 82.9% |
| 50 | 50 | 56.2% | 83.3% |
| 100 | 95 | **89.3%** | 85.7% |
| 200 | 107 | 91.7% | 85.7% |

The simulated attack is intentionally rich in the anomaly queue, so this is an upper-bound-style demonstration of anomaly alert → analyst confirmation → supervised retraining.

For the later test window, a **7-day as-of mature-label** view gets **0% novel recall**, while an invalid instant-label oracle gets **88.4%**. The oracle quantifies label-latency leakage and is not deployable.

## 7. Verification bias

Risk-triggered follow-up alone creates a labelled historical sample with **8.24% fraud prevalence** versus **1.13%** in the full historical population. Known-fraud PR-AUC is **0.584** and mule-cashout recall **11.1%**, versus **0.646 / 28.6%** with full labels.

Adding a random audit lane improves coverage in the simulator. The project does not claim a specific audit percentage is optimal in production.

## 8. Base-rate and calibration sensitivity

Holding measured TPR/FPR fixed, expected supervised-alert precision falls to about **3.0%** at **0.1% fraud prevalence**, versus **23.8%** at 1%. Synthetic observed precision must not be copied into production staffing estimates.

Under a controlled 0.10% prior-shift simulation, prior correction moves mean predicted risk from **0.679% to 0.085%** and Brier from **0.001665 to 0.000855**. At 2% prevalence, correction slightly worsens Brier, showing that label shift alone does not explain attack-driven concept drift.

## 9. Fraud Ops queue saturation in the 120k simulator

The reference review + exploration policy creates about **5.42 candidates/hour**. In the stated queue scenarios:

- 4 reviews/hour is overloaded at 1x traffic;
- 6 reviews/hour meets the four-hour wait proxy at 1x traffic;
- 8 reviews/hour becomes overloaded at 1.5x traffic.

A backlog-aware admission controller keeps the queue capacity-feasible by rejecting more candidates as traffic rises. At 1x traffic candidate acceptance is **88.3%**; at 1.5x it is **67.8%**; at 4x it is **27.2%**, while novel-fraud recall falls to **20.5%**.

The staffing rates, traffic multipliers and four-hour proxy are stress-test assumptions.

## 10. Recipient investigation audit: negative result

PaySim does not provide a confirmed mule-account label. Standalone strict prior-step recipient features are therefore evaluated only as investigation signals.

On future test, recipient fan-in AUC is about **0.493**, recipient-intensity AUC about **0.467**, and all validation-selected standalone recipient thresholds recover **0% future fraud**. Recipient activity is not promoted as a PaySim mule proxy.

## 11. Reproducibility and SQL correctness

- equal-timestamp Python events are processed as a batch;
- SQLite point-in-time SQL is executed and checked against Python features;
- PaySim uses strict prior-step DuckDB windows;
- a stable non-label `event_key` makes split loading and capacity tie-breaking deterministic;
- independent full PaySim benchmark and monitoring workflows reproduce the selected relational model to numerical precision;
- CI runs unit tests, a deterministic smoke workflow and report build.

PaySim raw data, materialised feature tables and fitted models remain runner-local. Only aggregate outputs are retained.
