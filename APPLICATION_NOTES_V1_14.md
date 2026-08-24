# Application notes — v1.14 backlog TTL frontier

Use this only as **portfolio / benchmark evidence**. PaySim is synthetic; do not describe the workload, value or delay figures as Moniepoint production measurements.

## Best CV wording

For roles emphasising Fraud Ops, ML lifecycle, model governance or decision systems:

> Built a causal fraud-review backlog lifecycle on the full 6.36M-row PaySim benchmark and predeclared an age-expiry frontier across six TTLs; showed that a 20-step TTL cut model-refresh rescoring workload by 60.3% and mean review delay from 8.35 to 3.84 dataset steps, but rejected it because precision fell 2.76 pp and breached the frozen 2 pp detection guardrail.

This is stronger than claiming the TTL improved the system, because the engineering rule was fixed before the result and the candidate was not accepted after it missed the guardrail.

A shorter alternative:

> Audited backlog ageing across model releases on a 6.36M-row fraud benchmark, holding review capacity fixed at 50/10k; no finite TTL passed predeclared detection and workload guardrails, so retained the unbounded reference rather than tuning thresholds post hoc.

## Interview story

The key question is not “which TTL has the best metric?” It is:

**How much review latency and release-time rescoring can be removed before the investigation queue changes too much?**

The baseline `rescore_pending` lifecycle has:

- 617 reviews;
- 52.67% precision;
- 19.65% fraud recall;
- 63.39% fraud-value recall;
- balanced H-mean 0.3000;
- mean review delay 8.35 PaySim steps;
- 150,847 total refresh rescoring operations across the two model releases.

The TTL grid was frozen at `0, 5, 10, 20, 40, infinite` before reading full-data results. A finite TTL had to lose no more than 2 percentage points separately on precision, fraud recall and fraud-value recall, while reducing refresh rescoring by at least 50%.

### Why TTL 20 was not accepted

TTL 20 looks attractive operationally:

- total refresh rescores: 150,847 → 59,845 (**−60.3%**);
- mean review delay: 8.35 → 3.84 steps;
- fraud recall: 19.65% → 18.62% (**−1.03 pp**);
- fraud-value recall: 63.39% → 61.57% (**−1.82 pp**).

But precision changes from 52.67% to 49.92%, a **−2.76 pp** decline. That fails the frozen −2 pp precision guardrail.

Good phrasing:

> “TTL 20 was almost the operational compromise I wanted, but I had declared the detection guardrail before seeing the result. It saved about 60% of refresh rescoring and more than halved mean queue delay, yet precision missed the allowed margin. I kept the negative result instead of relaxing the threshold.”

### Why TTL 40 was also not accepted

TTL 40 preserves more of the queue:

- 95.79% queue overlap with infinite TTL;
- only 26 of 617 review assignments are replaced;
- fraud recall declines 0.97 pp;
- fraud-value recall declines 1.18 pp.

But precision declines 2.59 pp and refresh rescore workload falls only 42.0%, below the frozen 50% workload target.

So the frontier has a real gap: the tested TTLs do not simultaneously meet the quality and workload requirements.

## A useful engineering detail

At model release, infinite TTL requires rescoring:

- 42,204 pending cases at step 645;
- 108,643 at step 695.

TTL 20 reduces these to 8,484 and 51,361 respectively.

This is useful evidence that model deployment is not just model serialisation. A new scoring model can imply a substantial backlog migration/rescoring workload even when the new-arrival scoring path itself is cheap.

## Do not misuse the expiry counts

Finite TTLs eventually expire many unreviewed transactions. Do not say those are all “fraud cases lost because of TTL”. Under 50 reviews/10k, most transactions are never selected even under infinite TTL.

Use the realised 617-review queue comparison instead:

- TTL 20 retains 88.17% of the infinite-TTL queue and replaces 73 assignments;
- TTL 40 retains 95.79% and replaces 26.

The label composition of expired cases is a retrospective diagnostic, not the routing rule.

## Claim boundaries

Do not claim:

- real analyst SLA or review latency;
- real Moniepoint staffing or queue volume;
- prevented loss from PaySim transaction amount;
- that TTL 20 or TTL 40 should be deployed;
- statistical non-inferiority from the deterministic 2 pp guardrail;
- that the tested TTL grid exhausts all possible operational policies.

The safe conclusion is:

> “I treated backlog age and model-refresh workload as governed operating parameters. On the full PaySim benchmark, none of six predeclared TTLs simultaneously met the frozen detection and workload targets, so I did not promote a finite TTL.”
