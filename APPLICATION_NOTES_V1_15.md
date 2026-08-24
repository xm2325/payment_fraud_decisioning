# Application notes — v1.15 bounded priority backlog

Use these results only as **portfolio / benchmark evidence**. PaySim is synthetic; do not present the queue, workload, amount or delay figures as Moniepoint production measurements.

## Strong CV wording

For Fraud Ops, model lifecycle, ML platform or decision-science roles:

> Built a bounded causal fraud-review backlog on the full 6.36M-row PaySim benchmark; under a predeclared score-priority cap grid and fixed 50 reviews/10k capacity, the lowest tested non-zero 5k cap preserved the **exact same 617-review queue** as the unbounded reference while reducing model-refresh rescoring from **150,847 to 10,000 operations (−93.4%)**.

A more governance-oriented version:

> Compared age-based and score-priority backlog controls under frozen detection/workload guardrails; age TTLs failed the joint screen, while predeclared 5k/10k/25k priority caps passed without changing the realised investigation queue, demonstrating why model-release workload should be governed separately from predictive metrics.

Do not write “5k is the optimal/minimum production backlog”. It is only the lowest non-zero value in the predeclared tested grid.

## Interview story

### 1. Start from the operations failure, not the positive result

The unbounded causal `rescore_pending` lifecycle uses 617 reviews over future steps 595--743 and reaches:

- precision **52.67%**;
- fraud recall **19.65%**;
- fraud-value recall **63.39%**;
- balanced H-mean **0.3000**;
- mean review delay **8.35 PaySim steps**.

But it requires rescoring 42,204 pending cases at the first model refresh and 108,643 at the second: **150,847 refresh rescoring operations**.

v1.14 first tried an age TTL. No tested finite TTL passed both the frozen detection guardrails and the >=50% workload-reduction target. That negative result was retained.

### 2. Change the mechanism, not the guardrail

v1.15 asks whether the backlog should be bounded by **priority rather than age**.

The grid was fixed before reading full-data results:

`0, 5k, 10k, 25k, 50k, 100k, infinite` pending cases.

After each step earns and spends its new analyst capacity, any remaining backlog above the cap loses its lowest current-score cases. At a model refresh, all surviving pending cases are rescored by the newly released model.

The same v1.14 screen remains unchanged:

- precision no worse than −2 pp versus infinite;
- fraud recall no worse than −2 pp;
- fraud-value recall no worse than −2 pp;
- refresh rescoring reduced at least 50%.

### 3. The 5k/10k/25k rows pass without changing the realised queue

The surprising result is exact, not a rounded approximation.

For cap=5k, 10k and 25k:

- all 617 selected transactions are identical to the infinite-backlog set;
- queue overlap = **100%**;
- replacement count = **0**;
- precision, fraud recall and fraud-value recall deltas = **0**;
- aggregate delay metrics are unchanged.

The lowest tested non-zero cap, 5k, rescored only:

- 5,000 pending cases at step 645;
- 5,000 at step 695;
- **10,000 total**, versus 150,847 for infinite backlog.

That is a **93.37% reduction** in refresh rescoring workload on this benchmark.

Good phrasing:

> “I did not relax the guardrails after the TTL experiment failed. I changed the queue-control mechanism instead. A score-priority cap of 5,000—the lowest non-zero value in my predeclared grid—retained exactly the same 617 investigations as the unbounded backlog while cutting refresh rescoring by 93.4%.”

### 4. Explain why this is possible

Under 50 reviews per 10,000 transactions, the realised investigation set is extremely sparse: 617 reviews among 123,580 future transactions.

The infinite backlog stores a very large ranking tail that never receives analyst capacity. A 5k priority reservoir removes much of that tail while retaining every case that eventually enters the realised infinite-baseline review set.

This does **not** imply the evicted tail contains no fraud. Retrospectively, the 5k policy evicts many labelled fraud transactions. The relevant operational observation is that none of those evicted transactions belongs to the infinite policy's realised 617-review set on this horizon.

So do not say “5k loses no fraud”. Say **“5k preserves the realised investigation queue and its measured capture metrics relative to the infinite-backlog policy.”**

### 5. Do not overclaim the boundary

The study did not test caps between 0 and 5,000. Testing 1k/2k/3k after seeing the 5k result would be a new parameter-search experiment.

Good phrasing:

> “Five thousand is the lowest tested passing cap, not an estimated minimum sufficient backlog. I would need a new predeclared validation design before narrowing that boundary.”

## Claim boundaries

Do not claim:

- real Moniepoint data or analyst workload;
- 150,847 real production inference calls;
- real latency or SLA measured in PaySim steps;
- prevented loss from PaySim transaction amount;
- that 5k is the production-optimal or universally sufficient cap;
- that evicted labelled fraud cases are incremental misses caused by the cap;
- statistical non-inferiority from the deterministic 2 pp guardrails;
- deployment readiness from `operational_candidate=True`.

The safe conclusion is:

> “On the fixed PaySim lifecycle, score-priority backlog bounding was much more efficient than age-only expiry: the predeclared 5k/10k/25k caps passed the unchanged detection/workload screen, and the 5k row preserved the exact realised 617-review queue while reducing refresh rescoring by 93.4%.”
