# Application notes — v1.9 policy promotion gate

Use these results only as portfolio / benchmark evidence. PaySim is synthetic and the project does not measure Moniepoint production fraud, prevented loss, staffing impact or real label maturity.

## Strong CV wording

For roles that stress model governance, fraud operations or model lifecycle, the strongest v1.9 bullet is:

> Built an uncertainty-aware fraud policy promotion gate on the full **6.36M-row PaySim** benchmark, comparing refreshed versus frozen exact-capacity review queues with a paired time-block bootstrap and family-wise error control; rejected a candidate despite a **+20.2 pp fraud-value recall** point gain because its family-adjusted case-recall guardrail did not pass.

A more technical version is:

> Extended a three-cycle rolling-origin fraud pipeline with a pre-declared promotion rule across six candidate-vs-incumbent comparisons; used 2,000 paired circular 5-step bootstrap replicates with Bonferroni one-sided family control, requiring a >=2 pp primary gain plus non-inferiority guardrails before policy replacement.

Do not claim that the candidate reduced real financial loss or that the statistical gate is production-validated.

## Strong interview story

v1.8 produced an attractive late-period result. At the same 50 reviews per 10,000 transactions, cycle-3 value-first routing moved from `alpha=0.25` to `alpha=1.0` and increased fraud-value recall from **40.76% to 60.95%**, while the full-window totals showed the same **100% precision and 12.77% fraud-case recall**.

v1.9 asks whether that result is strong enough to replace the incumbent. The release rule was fixed before reading the v1.9 bootstrap result: six cycle-2/3 profile comparisons, family alpha 0.05, 2,000 paired circular 5-step bootstrap replicates, at least +2 pp on the profile primary metric, a positive family-adjusted lower bound, and -2 pp non-inferiority margins on non-primary metrics.

For cycle-3 value-first, the fraud-value recall delta is **+20.20 pp** and its family-adjusted one-sided lower bound is **+9.78 pp**. That part is strong. However, the paired time-block lower bound for fraud-case recall is **-2.98 pp**, outside the declared -2 pp guardrail. The correct decision is therefore **KEEP_INCUMBENT**.

Good phrasing:

> “The headline value gain survived family-adjusted uncertainty, but the case-recall guardrail did not. I kept the incumbent rather than changing the release rule after seeing the result.”

This is stronger than saying the alpha=1.0 policy was simply better. It demonstrates that metric selection, uncertainty and release criteria were separated from the attractive observed result.

## Why the full-window equality is not enough

The incumbent and cycle-3 value-first candidate both have 12.77% fraud-case recall over the entire test window, but they do not necessarily capture the same fraud cases at the same times. The paired block bootstrap resamples contiguous time segments and exposes that temporal composition risk. That is why a zero full-window recall delta can still have a negative lower bound.

Good phrasing:

> “Two policies can tie on aggregate recall yet have different temporal failure patterns. I used paired block resampling to avoid treating aggregate equality as operational equivalence.”

## Safe limits

Do not say:

- that the bootstrap proves future production performance;
- that the gate includes model-parameter or retraining uncertainty;
- that PaySim contains realistic chargeback or investigation-completion timing;
- that alpha=1.0 should replace the balanced default;
- that the 20.2 pp gain is prevented-loss improvement.

The bootstrap measures uncertainty in incremental realised outcomes for two frozen queues on a completed PaySim test window. The decision can only be made after that window's labels are treated as available.

## One-minute summary

“I built the project as a fraud decisioning lifecycle rather than just a classifier. After temporal training, calibration, routing selection and rolling refresh, I added a release gate for candidate routing policies. On full PaySim, a value-first candidate looked excellent on the final window: value recall improved by 20.2 percentage points at the same aggregate case recall. The family-adjusted block-bootstrap lower bound for the value gain was still +9.8 points, but the case-recall guardrail lower bound was -3.0 points, outside my pre-declared -2-point margin. So I kept the incumbent. The key point is that I did not change the governance rule after seeing a strong headline result.”
