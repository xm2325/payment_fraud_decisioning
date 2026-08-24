# Application notes — v1.10 dependence sensitivity

Use these results only as **portfolio / benchmark evidence**. PaySim is synthetic and does not provide real chargeback maturity, investigation-completion timestamps, production review capacity, or prevented-loss outcomes.

## Strong CV wording

For a fraud, risk, model-governance, or decision-science role, this is the strongest v1.10-specific bullet:

> Built an uncertainty-aware fraud-policy release gate on the full **6.36M-row PaySim** benchmark using paired time-block bootstrap inference and family-wise error control; a candidate improved fraud-value recall by **20.2 pp**, but I retained the incumbent because the release decision was sensitive to temporal-dependence assumptions rather than selecting the favourable bootstrap setting post hoc.

A more technical version is:

> Extended a rolling-origin fraud decisioning benchmark with paired circular block-bootstrap policy comparisons across **1/3/5/10-step** dependence assumptions; five of six candidate comparisons were robustly rejected, while the strongest value-first candidate changed from `KEEP_INCUMBENT` to `PROMOTE` only at 10-step blocks, so the frozen 5-step governance decision remained unchanged.

Do not claim that the candidate was deployed, that the 10-step bootstrap is the correct production dependence model, or that the 20.2 pp gain represents prevented loss.

## Interview story

The important point is not that one bootstrap setting gives a better answer. It is that the release conclusion itself became a quantity to stress-test.

The candidate in cycle 3 increased fraud-value recall from about **40.76% to 60.95%** at the same full-window fraud-case recall. Under the frozen v1.9 5-step block bootstrap, the family-adjusted value-recall lower bound remained strongly positive at **+9.78 pp**, but the fraud-case-recall lower bound was **-2.98 pp**, outside the fixed -2 pp non-inferiority margin. The release gate therefore returned `KEEP_INCUMBENT`.

v1.10 reran the exact same rule under block lengths 1, 3, 5 and 10. The candidate remained `KEEP_INCUMBENT` for 1, 3 and 5 steps, but changed to `PROMOTE` for 10 steps because the fraud-case-recall lower bound rose to **-1.84 pp**. The value-recall lower bound stayed positive under all four assumptions.

A strong explanation is:

> “The headline value gain was not the uncertainty problem. The weak point was the case-recall guardrail. When I changed only the time-dependence assumption, the release decision changed at a 10-step block length. I therefore labelled the result dependence-sensitive and kept the pre-registered 5-step decision. I did not choose the block length that gave the deployment answer I wanted.”

That shows three useful behaviours: separating statistical evidence from the release rule, checking dependence assumptions rather than treating bootstrap intervals as automatic truth, and refusing post-hoc parameter selection.

## What remains unresolved

The sensitivity audit does not identify the true temporal correlation length. With only the PaySim benchmark, choosing 1, 3, 5, or 10 steps as the production dependence model cannot be justified from real investigation processes.

It also does not include uncertainty from refitting the fraud model or recalibrator. The paired bootstrap conditions on the realised frozen queues and their completed-window outcomes.

A production version would need real event timestamps, label-maturity timestamps, investigation completion, operational batch/campaign structure, and enough completed monitoring periods to estimate temporal dependence rather than selecting a block length by convention.

## Claims not to make

Do not say that v1.10 proves the value-first candidate is safe to deploy. Do not say the 10-step result supersedes the 5-step result. Do not describe the 20.2 pp gain as money saved. Do not imply PaySim has realistic chargeback or investigation timing.

The correct conclusion is: **the candidate has strong value-recall evidence, but its release decision is dependence-sensitive, so the incumbent remains the governed choice under the frozen v1.9 rule.**
