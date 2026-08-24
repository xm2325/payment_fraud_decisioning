# Application notes — v1.11 causal analyst capacity

Use these results only as **portfolio / benchmark evidence**. PaySim is synthetic; do not present its transaction amounts as prevented loss or its dataset steps as a real investigation service-level agreement.

## Recommended CV bullet

For fraud data science, risk, decision science, or model-governance roles:

> Built a causal analyst-capacity audit on the full **6.36M-row PaySim** benchmark, replacing whole-window score hindsight with a seen-so-far priority backlog; at 50 reviews per 10k transactions the causal queue reached **50.9% precision, 19.0% fraud recall and 64.1% fraud-value recall**, while explicitly measuring queue churn and review delay.

For roles that value methodological correction and model governance:

> Audited a fixed-capacity fraud-routing backtest for temporal hindsight and found that retrospective whole-window top-k overstated online performance; implemented two future-safe queue contracts with identical review budgets and relabelled the earlier **61.6% / 23.0% / 77.7%** precision/case-recall/value-recall result as a retrospective batch benchmark rather than an online claim.

The second bullet is often stronger in an interview because it shows that you tested your own evaluation assumptions and changed the claim when the evidence required it.

## Interview explanation

A concise explanation is:

> “My first exact-capacity evaluation was label-safe but not fully online. It ranked the whole future window and then took the top K, so an early transaction could lose a review slot to a transaction whose score would only arrive later. I kept that result as a retrospective upper benchmark and added two causal queues. A strict current-step queue dropped sharply, while a seen-so-far backlog recovered much of the performance without seeing future scores. At 50 reviews per 10k, the backlog achieved 50.9% precision, 19.0% case recall and 64.1% value recall, with 88.2% queue overlap against the retrospective batch queue.”

If asked why the backlog is more realistic than the current-step-only rule:

> “Analyst work does not have to forget an alert merely because it was not selected in its arrival micro-batch. The backlog lets previously observed transactions remain eligible for later capacity. It still makes every decision using only scores observed by that time.”

If asked about the cost of backlog flexibility:

> “I measured delay rather than treating the backlog as free. At 50 reviews per 10k, mean delay was 7.72 PaySim steps, p90 was 28 and the maximum was 92. Those are dataset-step diagnostics, not a production SLA, because PaySim does not model a real analyst service process.”

## Which number should be used now?

For a general causal Fraud Ops claim, prefer the **seen-so-far backlog** result at 50 reviews/10k:

- precision: **50.89%**;
- fraud-case recall: **18.98%**;
- fraud-value recall: **64.10%**;
- queue overlap with retrospective batch: **88.17%**;
- review replacement rate: **11.83%**;
- mean review delay: **7.72 PaySim steps**;
- p90 review delay: **28 steps**.

The older **61.59% precision / 22.97% fraud recall / 77.67% fraud-value recall** numbers remain valid only as a **retrospective whole-window batch benchmark** under the same fixed total capacity.

The strict current-step-only result, **23.01% precision / 8.59% fraud recall / 27.42% fraud-value recall**, is useful as a low-latency stress case. Do not present it as the only possible causal queue design.

## Why this matters for the earlier v1.9/v1.10 evidence

The v1.9 promotion bootstrap and v1.10 dependence sensitivity are retrospective completed-window policy audits. They remain useful for comparing frozen policies after outcomes mature, but they should not be described as a literal online queue deployment simulation.

v1.11 separates two questions:

1. **retrospective policy evaluation:** after a completed period, how did two frozen rankings compare under a common capacity budget?
2. **causal queue execution:** while transactions are arriving, which items could actually have been sent to analysts without knowledge of later scores?

Keeping these questions separate makes the project evidence easier to defend.

## Claims not to make

Do not claim that:

- the whole-window top-k result is online routing;
- the earlier result contained fraud-label leakage — the issue is future-score hindsight in capacity allocation;
- PaySim review delay maps directly to hours or a real analyst SLA;
- the backlog is a complete production queue simulator;
- synthetic transaction amount is prevented loss or money saved;
- 50 reviews per 10k is a measured Moniepoint staffing level.

The safe conclusion is: **the model and routing policy remain useful, but causal capacity execution materially lowers the retrospective headline; a seen-so-far backlog recovers much of that gap at a measurable delay cost.**
