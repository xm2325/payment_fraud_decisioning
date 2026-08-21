# Payment Fraud Decisioning & Early-Warning Workbench

A fraud data-science portfolio project focused on **temporal modelling, point-in-time SQL, Fraud Ops capacity, emerging-fraud discovery, delayed labels and decision-policy governance**.

The project was built to practise the kind of work expected in a payments/fraud Data Scientist role: not just ranking transactions, but deciding **which cases to review under a constrained analyst budget, what happens when fraud load changes, and which conclusions survive out-of-time testing**.

## Evidence boundaries

There are two data layers and they are deliberately kept separate:

- a transparent **120,000-transaction synthetic stream** used for controlled experiments on emerging attacks, label delay, verification bias, analyst feedback and queue saturation;
- the complete public **6,362,620-row PaySim** mobile-money simulator used as a reproducible external benchmark through GitHub Actions.

Neither source is Moniepoint data. No result in this repository is a production fraud rate, prevented-loss claim, staffing recommendation or measured customer impact.

## v1.6 verified PaySim decisioning path

GitHub Actions downloads the canonical PaySim Parquet shards and hard-fails unless it sees **6,362,620 transactions, 8,213 fraud cases and steps 1–743**. DuckDB builds strict prior-step features; same-step transactions cannot see one another. The split is train 1–445, validation 446–594 and untouched future test 595–743.

### 1. Dataset audit before model headline

PaySim old/new-balance derivatives make fraud nearly perfectly separable (PR-AUC around **0.995**), so those fields are sensitivity-only. The portfolio headline is the **balance-free relational model**.

Standalone recipient/fan-in signals are also retained as a negative result: their future AUC is around random and they produce **0% fraud recall** at the audited narrow thresholds. PaySim has no confirmed mule-account label, so the repository does not claim mule detection.

### 2. Exact review capacity, not a brittle scalar threshold

LightGBM scores contain large ties. A scalar `score >= threshold` can therefore jump over a narrow alert budget or leave much of it unused. v1.4 changed the operational contract to **exact top-k routing** with a stable non-label event key as the tie-breaker.

At exactly **50 alerts per 10,000 future transactions**:

| ranking policy | precision | fraud-case recall | fraud-value recall |
|---|---:|---:|---:|
| relational `P(fraud)` | **64.34%** | **24.00%** | 71.67% |
| validation-selected compromise `P(fraud) × amount^0.25` | 61.75% | 23.04% | **77.70%** |
| `P(fraud) × amount` | 56.40% | 21.04% | 76.96% |
| simple amount/type rule | 42.46% | 15.84% | 70.43% |

The compromise exponent **0.25** is selected on validation only from the pre-specified grid `{0, 0.25, 0.5, 0.75, 1}`. Case-first, balanced and value-first validation objectives all chose the same exponent, so v1.5 reports **one compromise ranker**, not three artificial profiles.

Relative to pure risk ranking, the frozen `alpha=0.25` future queue trades about **0.97 percentage points of case recall** and **2.59 points of precision** for about **6.03 points of fraud-value recall**. It also outperforms the `alpha=1` endpoint on all three reported future metrics.

### 3. Score-load surge-capacity stress test

A fixed review budget fails when fraud arrival accelerates. v1.6 therefore tests a prospective capacity-flex rule that uses **model-score load only**:

- fit the validation top-0.5% score-tail reference;
- keep a baseline **50 alerts/10k** capacity;
- under the reference scenario, flex to **100 alerts/10k** when future score-tail load reaches **1.5×** the validation reference;
- evaluate future fraud labels only after the capacity decision.

The full PaySim run measured score-tail multipliers of **1.502×, 1.661× and 2.407×** across the three future windows, so the reference 1.5× policy triggered all three.

| window | fixed value recall | 100/10k surge value recall | fixed case recall | surge case recall | surge precision |
|---|---:|---:|---:|---:|---:|
| 1 | 75.45% | 85.06% | 23.55% | 30.62% | 39.86% |
| 2 | 82.30% | 89.55% | 27.11% | 34.98% | 28.64% |
| 3 | **40.76%** | **81.43%** | **12.77%** | **25.54%** | **99.30%** |

The final window is the clearest capacity-saturation example: doubling the stress-test review budget nearly doubles case recall and fraud-value recall while precision stays about 99%. This does **not** mean extra analysts are instantly available; it quantifies the coverage upside *if* flex capacity exists.

The 1.5× trigger is intentionally not treated as optimal. It is almost exactly crossed in window 1. v1.6 therefore reports a pre-declared **1.5 / 2.0 / 2.5** governance sensitivity: based on the observed score loads, 1.5× triggers all three windows, 2.0× triggers only the final window and 2.5× triggers none. Future labels are not used to choose among these thresholds.

See `PAYSIM_SURGE_CAPACITY.md`, `PAYSIM_ROUTING_PROFILES.md`, `PAYSIM_MONITORING.md` and `PAYSIM_BENCHMARK.md` for the contracts and caveats.

## Controlled synthetic stress tests

The internal 120k stream is useful because the attack mechanism and label availability are known exactly:

- point-in-time velocity/history raised known-fraud PR-AUC from **0.597 to 0.640** and fraud-value recall from **78.1% to 86.1%**;
- the supervised model had **0% recall** on a deliberately unseen shared-device attack, while a label-free tail detector reached **93.3% recall at 0.86% legitimate flag rate**;
- at 200 reviews/10k, an 80/20 exploit-explore queue preserved **80.9% fraud-value recall** while adding **40.2% new-attack recall**;
- 100 anomaly-ranked discovery reviews produced 95 simulated novel confirmations and raised later novel recall from 0% to **89.3%** after retraining;
- a 7-day mature-label view remained at **0% novel recall**, while an invalid instant-label oracle reached **88.4%**, demonstrating temporal leakage;
- queue stress shows that model quality and analyst capacity are different failure modes: the reference stream creates about **5.42 review candidates/hour**, and fixed staffing assumptions can become overloaded even when the classifier itself has not failed.

These are method stress tests, not real-world incidence or intervention estimates.

## Architecture

```text
transaction stream
   -> strict point-in-time Python / SQL features
   -> calibrated supervised risk -----------------> risk-ranked queue
   -> amount-aware governed prioritisation -------> value-aware queue
   -> label-free anomaly channel -----------------> exploration queue
   -> exact review-capacity contract
   -> score-load / backlog monitoring
   -> approve / review / block + reason codes
   -> analyst outcome / mature fraud label
   -> as-of retraining + feedback loop
```

## Engineering controls

- DuckDB full-data point-in-time feature materialisation.
- SQLite/Python parity test for equal-timestamp history semantics.
- Stable non-label event keys for deterministic tie-breaking.
- pandas 3 timestamp-resolution regression test.
- FastAPI policy endpoint, Docker, pytest and GitHub Actions.
- Full PaySim workflows publish **aggregate results only**; raw Parquet, feature materialisations and fitted models stay runner-local.
- Bot result publishing rebases/retries so concurrently finishing full-data workflows do not overwrite one another.

## Run locally

```bash
python -m pip install -r requirements.txt
pytest -q
FRAUD_N=30000 python scripts/run_all.py
python scripts/build_report.py
uvicorn fraud_decisioning.api:app --app-dir src --reload
```

The 6.36M-row workflows are designed for GitHub Actions or another networked environment with enough temporary disk rather than the default local smoke run.

## Start here

- `CASE_STUDY.md` — concise project narrative.
- `TAKE_HOME_WALKTHROUGH.md` — SQL/Python/case-study interview walkthrough.
- `RESULTS.md` — detailed synthetic and PaySim evidence.
- `PAYSIM_BENCHMARK.md` — full-data modelling contract.
- `PAYSIM_MONITORING.md` — threshold vs capacity monitoring semantics.
- `PAYSIM_ROUTING_PROFILES.md` — validation-selected `alpha=0.25` routing compromise.
- `PAYSIM_SURGE_CAPACITY.md` — score-tail flex-capacity stress test.
- `APPLICATION_NOTES.md` — safe CV/interview wording and claims not to make.
- `MODEL_CARD.md` / `SYSTEM_DESIGN.md` / `DATA_PROVENANCE.md` — deployment and evidence boundaries.

## Claims deliberately not made

Do **not** read the benchmark as evidence of real Moniepoint performance. The repository does not claim real prevented loss, production fraud prevalence, a real A/B treatment effect, real analyst throughput, confirmed mule accounts, real staffing SLAs or live customer-friction impact. PaySim and the internal stream are synthetic; their purpose is to make modelling and decision-policy assumptions testable.
