# PaySim threshold and recipient monitoring — v1.4

This document defines the monitoring contract for the full 6.36M-row PaySim benchmark. It deliberately separates **prospective operating decisions** from **retrospective diagnostics**.

## 1. Operating-threshold contract

The reference balance-free model is `transaction_plus_relational`. Model selection and the operating threshold use validation data only.

1. Train on steps 1--445.
2. Calibrate and select the legitimate-alert operating point on validation steps 446--594.
3. Lock that threshold.
4. Apply the same threshold to future steps 595--743.
5. Measure realised future alert load, precision, fraud-case recall and fraud-value recall without using future labels to alter the operating point.

The target validation legitimate flag rate is **0.1% (10 legitimate alerts per 10,000 legitimate transactions)**. Because a fixed score threshold can drift away from that budget later, the monitoring layer reports both the realised rate and its multiplier relative to the target.

## 2. Budget-drift measures

`results/paysim_monitoring/threshold_budget_drift.csv` reports:

- locked score threshold;
- legitimate flag rate;
- legitimate alerts per 10,000 legitimate transactions;
- multiplier versus the 0.1% validation target;
- multiplier versus the realised validation rate;
- precision and fraud-case recall;
- fraud-value recall;
- score distribution p50 / p90 / p99 / p99.9.

`future_budget_windows.csv` applies the same threshold to three contiguous future step windows. This distinguishes persistent capacity drift from a short-lived spike.

### Proposed governance guardrail

For a real deployment, an example governance rule would be to investigate when a locked threshold produces a sustained legitimate-alert budget above **1.5x the agreed target**, particularly if the score tail or transaction mix shifts at the same time. The **1.5x value is a proposed guardrail, not a PaySim-derived optimum and not a Moniepoint policy**.

The response to an alert should be investigation first: score-distribution shift, feature/data quality, transaction mix, fraud prevalence, calibration and intervention capacity. Automatic re-thresholding is not assumed to be safe.

## 3. Post-hoc budget-matched threshold

`future_budget_rethreshold_oracle.csv` asks a deliberately retrospective question: what threshold would have been required, after seeing the future labels, to restore the 0.1% legitimate-alert budget?

That threshold is marked **diagnostic only**. It can quantify how much the operating point moved, but it cannot be reported as a prospective model result because it uses future labels.

A production analogue would require a governance-approved recalibration/re-thresholding process based only on information available at the time, with mature labels and holdout validation.

## 4. Recipient investigation signals

PaySim does not contain a confirmed mule-account label. The v1.4 recipient audit therefore does **not** claim mule detection. It evaluates whether strict prior-step recipient activity could be useful for analyst investigation.

Signals include:

- prior 24-hour recipient fan-in;
- prior 7-day recipient transaction count;
- prior 24-hour recipient amount;
- prior 7-day approximate unique senders;
- a simple label-free recipient-intensity combination.

Each signal threshold is selected using the validation legitimate distribution and then locked for future evaluation. The output is `recipient_signal_audit.csv`.

If these standalone signals are weak, that is retained as negative evidence rather than hidden: it would imply that recipient activity alone is not a defensible mule proxy in PaySim, even if nonlinear relational features contribute inside the supervised model.

## 5. Point-in-time boundary

PaySim records time as hourly `step`, not exact within-hour event order. Recipient and relational features therefore use only steps strictly before the current step. Transactions in the same hour are treated as simultaneous and cannot use one another as history.

This avoids same-step leakage at the cost of discarding potentially available within-hour sequencing that PaySim does not provide.

## 6. Evidence boundary

PaySim is synthetic mobile-money data. Monitoring results demonstrate temporal evaluation, operational-budget reasoning and investigation workflow design. They are not production fraud rates, real customer-friction estimates, confirmed mule-account findings or saved-money claims.
