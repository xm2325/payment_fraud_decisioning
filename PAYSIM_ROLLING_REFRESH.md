# Full PaySim rolling-refresh audit — v1.8

This audit extends the v1.7 one-shot temporal lifecycle into a rolling-origin, as-of refresh test on the canonical **6,362,620-row PaySim** benchmark. It asks a specific operational question: after calibration drift is observed, does periodic model/calibration/routing refresh improve the next unseen period compared with leaving the v1.7 system frozen?

PaySim is synthetic mobile-money data. It has fraud labels but no real investigation completion time, chargeback maturity time or analyst-confirmation delay. The rolling strategy below is therefore an **as-of upper-bound under labels being available by the next refresh**, not evidence for a production delayed-label process.

## Temporal contract

The schedule is determined only from ordered PaySim steps. Labels and performance do not move any boundary.

| Cycle | Model training | Calibration only | Routing-policy selection only | Next test |
|---:|---|---|---|---|
| 1 | 1–445 | 446–519 | 520–594 | 595–644 |
| 2 | 1–495 | 496–569 | 570–644 | 645–694 |
| 3 | 1–545 | 546–619 | 620–694 | 695–743 |

Each refresh keeps **74 calibration steps** and **75 policy-selection steps** immediately before the next test origin. The model-training history expands by 50 steps per cycle. A cycle can use labels from completed earlier periods, but it cannot use labels from its own test period or any later period.

Cycle 1 is the v1.7 lifecycle with its first 50-step future window. Two strategies are then compared:

- `frozen_v1_7`: keep the cycle-1 model, sigmoid calibrator and routing profiles unchanged for all later tests;
- `rolling_refresh`: refit the model on the expanding training history, refit the calibrator on the declared calibration window, and select routing alpha only on the immediately preceding policy window.

The fixed feature family is `transaction_plus_relational`. Routing uses the pre-specified family

`priority = P(fraud) × (amount / policy_median_amount)^alpha`,

with `alpha ∈ {0, 0.25, 0.5, 0.75, 1}` and an exact capacity of **50 reviews per 10,000 transactions** for policy selection.

## Probability results: refresh helps one period and fails to anticipate the next base-rate jump

| Test cycle | Actual fraud rate | Frozen mean risk | Rolling mean risk | Frozen Brier | Rolling Brier | Frozen PR-AUC | Rolling PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1: 595–644 | 1.301% | 2.388% | 2.388% | 0.01215 | 0.01215 | 0.3303 | 0.3303 |
| 2: 645–694 | 0.818% | 2.495% | **1.058%** | 0.00957 | **0.00665** | 0.3175 | 0.3174 |
| 3: 695–743 | **3.863%** | 3.133% | **1.152%** | **0.02561** | 0.03015 | 0.5272 | 0.5331 |

Cycle 2 shows the intended benefit of recent-label recalibration: the frozen system materially over-predicts risk, while the rolling system moves mean risk much closer to the observed base rate and reduces Brier loss by about **30.5%**.

Cycle 3 shows the failure mode. Fraud prevalence jumps from below 1% in the prior periods to **3.863%**. The rolling calibrator was fitted on an earlier 0.641% calibration period and then evaluated after this jump, so it under-predicts badly. Its Brier loss is worse than the frozen v1.7 calibrator even though ranking PR-AUC is slightly higher.

This result rejects a simple claim that scheduled recalibration automatically solves temporal calibration drift. A recent calibrator can still lag a sudden base-rate shift. In a real system, refresh cadence, mature-label availability, prior/base-rate shift checks and escalation rules would need separate monitoring.

## Balanced routing: periodic refresh does not improve the default queue

The robust `balanced` policy remains **alpha=0.25 in all three cycles**. At 50 reviews/10k:

| Test cycle | Frozen precision | Rolling precision | Frozen case recall | Rolling case recall | Frozen value recall | Rolling value recall |
|---:|---:|---:|---:|---:|---:|---:|
| 595–644 | 61.32% | 61.32% | 23.55% | 23.55% | 75.45% | 75.45% |
| 645–694 | 42.04% | 42.04% | 25.64% | 25.64% | 81.74% | 81.54% |
| 695–743 | 100.00% | 100.00% | 12.77% | 12.77% | 40.76% | 40.76% |

The rolling model therefore gives **no consistent improvement to the default balanced queue**. The cycle-2 value-recall change is slightly negative (−0.19 percentage points), and cycle 3 is unchanged.

This separation is useful: probability calibration can change strongly while exact top-k routing remains almost unchanged because ranking is the operational contract and sigmoid calibration is monotone.

## Value-first routing: one large late-period gain, but not a stable general gain

The policy-selection windows choose:

- cycle 1: value-first `alpha=0.25`;
- cycle 2: value-first `alpha=0.5`;
- cycle 3: value-first `alpha=1.0`.

In cycle 2, moving to `alpha=0.5` does not help the next test: value recall changes from **81.74% to 81.52%**, with small losses in precision and case recall.

In cycle 3, however, the policy window selects `alpha=1.0`, and the next test has a strong value-selection effect:

| Cycle 3 value-first strategy | Precision | Case recall | Fraud-value recall |
|---|---:|---:|---:|
| Frozen v1.7, alpha=0.25 | 100.00% | 12.77% | 40.76% |
| Rolling selection, alpha=1.0 | 100.00% | 12.77% | **60.95%** |

The rolling value-first queue gains **20.20 percentage points** of fraud-value recall while admitting the same number of fraud cases and no legitimate cases. This means it selected higher-value fraud cases within a capacity-saturated period.

The project does **not** promote `alpha=1.0` to a new default. The gain occurs in one late synthetic period, while the preceding cycle slightly worsens. The defensible result is that a separately governed value-first objective can adapt differently from the stable balanced queue, and its benefit must be checked out of sample.

## What v1.8 changes in the project story

v1.7 established clean separation between model fitting, probability calibration, routing-policy selection and future evaluation. v1.8 tests what happens after the first future window is observed.

The evidence now supports four distinct statements:

1. **Calibration and ranking are different contracts.** Recalibration can materially change absolute risk without changing the top-k queue much.
2. **Recent-label recalibration is not automatically safer.** It helps in cycle 2 and then lags a sharp cycle-3 prevalence increase.
3. **The balanced routing rule is stable.** `alpha=0.25` is reselected in every cycle and its next-window queue metrics are essentially unchanged.
4. **A value-first policy can react differently.** In the final period, policy-selected `alpha=1.0` materially raises captured fraud value at the same case count, but this is not consistent enough to make it the default.

## Reproducibility

The full workflow is `.github/workflows/paysim-rolling-refresh.yml`. It downloads the same canonical PaySim parquet shards used by the other full-data audits, verifies **6,362,620 transactions / 8,213 fraud cases / steps 1–743**, materialises strict prior-step DuckDB features, runs all three rolling cycles, uploads the complete result set, and on a `main` push writes aggregate CSV/JSON results to `results/paysim_rolling_refresh/`.

Primary outputs are:

- `cycle_contract.csv` — auditable as-of boundaries;
- `probability_diagnostics_by_cycle.csv` — calibration and ranking diagnostics;
- `selected_profiles_by_cycle.csv` — validation-only routing choices;
- `policy_robustness_by_cycle.csv` — worst-window policy evidence;
- `routing_frontier_by_cycle.csv` — exact-capacity results;
- `routing_comparison_50_per_10k.csv` — frozen versus rolling comparison;
- `summary.json` — machine-readable contract and interpretation limits.

## Claim boundaries

Do not describe this audit as real-time Moniepoint retraining, measured financial impact or a validated delayed-label production schedule. PaySim does not supply the required label-maturity process. The controlled 120k layer remains the project evidence for explicit delayed labels and verification bias. A real deployment would need mature-label timestamps, investigation outcomes, intervention effects, traffic/capacity data and a governed trigger for recalibration or retraining.
