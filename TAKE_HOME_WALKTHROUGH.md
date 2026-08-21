# Fraud Data Science Take-home Walkthrough

This note turns the repository into a compact take-home style case: investigate the data, build point-in-time features, compare an interpretable rule with machine learning, choose the model on validation only, and explain what Fraud Ops should do next.

## 1. Start with the decision, not the model

For every transaction, the downstream action is one of:

- approve;
- step-up / challenge;
- manual review;
- block.

The modelling task is therefore not "maximise AUC". It is to rank risky transactions well enough to allocate a limited intervention budget while controlling legitimate-customer friction.

## 2. Data and split contract

The public PaySim benchmark is synthetic mobile-money data. The full run contains 6,362,620 transactions and 8,213 labelled fraud cases across steps 1--743.

The benchmark is time ordered:

- train: steps 1--445;
- validation: steps 446--594;
- future test: steps 595--743.

All rolling features stop at `1 PRECEDING`, so events in the same hour cannot see each other. Thresholds and model choice use validation only. The future period is read once for final evaluation.

## 3. SQL investigation questions

### A. Which recipients look structurally unusual?

Run `sql/paysim_recipient_investigation.sql` and rank recipients by prior 24-hour inflow, 7-day distinct senders and sender-recipient reuse. This is designed to surface mule-like fan-in and concentration patterns without using the fraud label online.

### B. How large are the main fraud typologies?

Run `sql/paysim_typology_sizing.sql` retrospectively on mature labels. The goal is to size loss/value by transaction type and amount band, not to turn labels into online features.

### C. Do simple rules already solve the problem?

Run `sql/paysim_rule_backtest.sql` and `sql/paysim_rule_incrementality.sql`. A rule is selected on validation and evaluated out of time, just like the model threshold.

### D. Is the operational threshold stable?

Run `sql/paysim_threshold_drift.sql` to compare train/validation/future score or amount distributions. This gives an analyst a direct way to reason about why a validation-derived false-positive budget may drift in production.

## 4. Baseline before ML

The supplied PaySim `isFlaggedFraud` rule is extremely precise but almost inactive on the future test: 8 alerts, 100% precision, 0.48% fraud recall and 1.56% fraud-value recall.

A stronger interpretable baseline uses transaction type plus amount. Its amount threshold is selected on validation to target a 0.1% legitimate flag rate. On the future test it reaches:

- precision: 46.6%;
- fraud recall: 12.1%;
- fraud-value recall: 62.2%;
- legitimate flag rate: 0.188%;
- alerts: 429.

This is the correct baseline for asking whether ML adds value.

## 5. Balance-free model selection

Three candidates are compared on validation PR-AUC only:

| Candidate | Validation PR-AUC |
|---|---:|
| transaction only | 0.2775 |
| transaction + history | 0.2733 |
| transaction + relational history | **0.3252** |

The relational model is therefore selected before the future test is examined.

Its future-test results are:

- PR-AUC: 0.3530;
- precision: 60.1%;
- fraud recall: 25.9%;
- fraud-value recall: 80.7%;
- legitimate flag rate: 0.233%;
- alerts: 712.

Relative to the simple amount/type rule, this roughly doubles case recall while raising precision and fraud-value coverage. That is a much more defensible statement than quoting the simulator-balance model's near-perfect score.

## 6. Why fraud-value recall needs a warning label

PaySim fraud value is highly concentrated:

- top 10% of fraud cases = 55.1% of fraud value;
- top 25% = 82.4%;
- top 50% = 95.3%.

So a model can obtain a high fraud-value recall by preferentially finding large fraud cases. The project therefore reports case recall, precision, legitimate flags, an amount-based rule baseline and fraud-value concentration together.

## 7. Why the balance model is not the headline

Adding old/new balance derivatives produces validation PR-AUC 0.9923 and future-test PR-AUC 0.9950. In PaySim this is evidence of simulator-specific accounting separability, not evidence that production fraud is almost perfectly predictable. The balance model remains a sensitivity audit only.

## 8. Fraud Ops recommendation

The practical recommendation is a layered system:

1. keep high-precision deterministic rules for obvious cases;
2. use the balance-free relational model to rank the broader review queue;
3. reserve a small exploration slice for anomaly-led discovery;
4. log analyst outcomes and mature labels separately;
5. monitor alert rate, precision, calibration, feature drift, typology mix and threshold stability;
6. retrain only after checking whether the change is label delay, prior shift, feature drift or a genuinely new attack pattern.

## 9. What I would request before productionisation

The public datasets do not contain enough information for a production claim. I would ask for:

- event-time device, IP, session and authentication signals;
- account age, KYC state and prior confirmed fraud history;
- payment rail / merchant / counterparty context;
- delayed chargeback or confirmed-fraud labels with maturity dates;
- exact intervention outcome: approved, challenged, reviewed, blocked;
- review capacity and service-level targets;
- legitimate-customer abandonment or complaint outcomes;
- feature availability timestamps to prevent training-serving leakage.

## 10. Interview summary

The main result is not "I built a fraud model". It is:

> I built a time-aware fraud decisioning benchmark, compared machine learning against deterministic rules, selected the balance-free champion only on validation, showed that relational features improve ranking and selectivity, and explicitly audited two misleading shortcuts: simulator balance leakage and highly concentrated fraud value.
