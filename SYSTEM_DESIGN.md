# Fraud decisioning system design — v1.3

```text
                              +--------------------+
transaction -> point-in-time | supervised model   | -> calibrated risk -> exploit lane
features                      +--------------------+
       |                      +--------------------+
       +--------------------> | anomaly detector   | -> early warning --> explore lane
                              +--------------------+
                                         |
                                         v
                              fixed Fraud Ops capacity
                                         |
                              reason codes + analyst review
                                         |
                         confirmed / mature outcome labels
                                         |
              +--------------------------+-------------------------+
              |                                                    |
       feedback retraining                                  monitoring
       with as-of cutoff                         immediate signals + mature labels
```

## 1. Point-in-time contract

Feature state is read before the current event enters history. Equal-timestamp events are processed as a batch so dataframe order cannot create same-time leakage. The SQLite reference query uses the same strict event-time rule, and a parity test checks Python against SQL.

## 2. Separate known-risk and discovery objectives

The supervised model targets known fraud with mature labels. Network-style features did not improve the supervised champion in the 120k ablation, so v0.9 does not add them merely because they sound fraud-specific. They remain available to anomaly scoring and analyst reason codes.

The anomaly channel is not used as a fraud probability. It exists to surface behaviour outside historical support before labels mature.

## 3. Capacity-aware Fraud Ops routing

Analyst capacity is expressed as reviews per 10,000 transactions. The fixed two-lane policy reserves 80% for high supervised risk and 20% for label-free exploration. Sensitivity from 0-40% exploration is reported rather than tuning the share on future novel-fraud truth.

## 4. Learning loop

The feedback simulation splits the post-attack period into discovery (days 48-53) and later evaluation (days 54-59). The anomaly channel ranks discovery cases; analyst-confirmed outcomes are added to training; the supervised model is then re-estimated and evaluated only on the later block. This makes explicit how anomaly detection can generate training signal rather than remain a permanent parallel rule.

## 5. Label-maturity contract

Scheduled retraining must use an `as_of_time`. Outcomes whose confirmation timestamp is later than `as_of_time` are not available for training. The oracle comparison exists only to quantify temporal leakage.

## 6. Verification coverage

Investigation policy can determine which labels are ever observed. v0.9 therefore includes a verification-bias stress test and a random audit lane. This is separate from anomaly exploration: one supports representative label coverage; the other seeks emerging risk.

## 7. Policy assumption audit

Validation policy selection is repeated under four intervention/cost scenarios. A future-test optimum is computed only retrospectively to report policy regret, never to choose the deployed threshold pair. This separates threshold selection from sensitivity auditing.

## 8. Base-rate planning

Ranking metrics and TPR/FPR do not determine analyst workload by themselves. Expected precision is recalculated across assumed fraud prevalences. Production planning would use observed deployment prevalence by market/product/segment, not the simulator prevalence.

## 9. Monitoring

Immediate signals: traffic volume, score distribution, alert rate, anomaly alert rate, queue composition and latency. Mature-label signals: precision, recall estimates, fraud-value capture, calibration and typology mix. A score PSI near zero does not suppress an investigation when label-free anomaly alerts change materially.

## 9. Probability transport across markets or products

Calibrated probabilities are not assumed to transport unchanged when fraud prevalence changes. v0.9 includes a prior-probability adjustment layer for controlled label-shift sensitivity. It is only valid when class-conditional score distributions are sufficiently stable; failure at the 2% stress point is kept as evidence that attack-driven concept drift needs richer recalibration or model updates.

## 10. Queue health is a deployment guardrail

Approve/review/block and exploration rules create a stream of analyst work. v0.9 aggregates review candidates hourly and simulates separate exploit/explore service lanes with spare-capacity sharing. Monitoring should include arrival rate, utilisation, backlog, wait time and lane mix. During demand spikes the safe action may be to change thresholds, temporarily resize exploration, add capacity or degrade to a simpler policy rather than let backlog grow without bound.

## v1.0 capacity-control loop

The operational path now distinguishes **candidate generation** from **review admission**. Static model/anomaly thresholds can generate more cases than Fraud Ops can process. The admission controller applies an hourly capacity budget, keeps a governance-defined long-run exploration reservation with fractional tokens carried across hours, uses the calibrated model score to rank exploitation cases and the label-free tail score to rank exploration cases, and spills unused capacity between lanes.

```text
model review band ----> exploit candidates --\
                                           > capacity-aware admission -> analyst queue
tail anomalies -------> explore candidates --/            |
                                                         backlog / SLA monitor
```

This is intentionally separate from model retraining. During a traffic spike, changing the admission threshold is an operational control; retraining the fraud model is justified only by model/label evidence.


## 11. External benchmark governance

The full PaySim path is intentionally separate from the 120k controlled stream. The external benchmark uses a canonical row/fraud/step audit, strict prior-step DuckDB windows and validation-only model selection among balance-free candidates. Relational features are retained only if the validation evidence justifies them; simulator balance derivatives are reported as sensitivity because they create near-perfect separability. A simple amount/type rule and fraud-value concentration check prevent a high value-recall number from being read as model skill without context.
