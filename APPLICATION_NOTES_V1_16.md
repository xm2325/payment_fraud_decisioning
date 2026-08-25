# Application-safe evidence — v1.16

## Strong CV bullet

Built a causal fraud-operations backlog stress test on the full 6.36M-row PaySim benchmark, showing that a fixed 5,000-case score-priority pending reservoir preserved the exact realised review path versus an infinite backlog across 10/25/50/100 reviews per 10k while reducing model-refresh rescoring by about 93%.

## More detailed CV / portfolio version

Extended a rolling fraud-decisioning benchmark from aggregate queue metrics to pathwise release validation: under identical models and score ordering, verified that a predeclared 5,000-case bounded priority backlog selected the same cases at the same review steps as an infinite backlog across four analyst-capacity levels (123-1,235 reviews), with zero queue displacement and ~93.3%-93.4% lower refresh-time rescoring.

## Interview explanation

The previous version showed that a 5,000-case pending cap produced the same final 617 investigations as an infinite backlog. I did not treat that as enough evidence, because the same final set could hide different investigation timing or could be specific to one analyst-capacity assumption.

I froze the 5,000-case cap and reran the causal queue at 10, 25, 50 and 100 reviews per 10,000 transactions. I compared not only the final selected masks but also each case's review step and the cumulative queue path. All four capacities were pathwise identical to the infinite queue: 0 divergent review steps and a minimum cumulative Jaccard of 1.0. At the highest tested capacity that meant the same 1,235 reviews at the same steps, while refresh rescoring remained capped at 10,000 cases in total instead of roughly 150,000.

The operational lesson is that a bounded priority reservoir can materially reduce release-time compute without changing realised analyst allocation when the removed tail is safely below the investigation frontier. The important part is validating pathwise equivalence rather than assuming it from aggregate metrics.

## Claim boundaries

Do not say:

- "5,000 is the minimum required backlog";
- "production workload is reduced by 93%";
- "the cap cannot lose fraud";
- "PaySim validates analyst SLA or prevented loss";
- "the result proves deployment readiness".

Safe wording:

- 5,000 was the lowest non-zero cap in the predeclared v1.15 grid;
- the v1.16 audit did not search below it;
- the ~93% reduction is refresh-time **rescoring volume in this PaySim benchmark**;
- pathwise preservation is established only for the tested synthetic horizon, fixed rolling systems and 10/25/50/100 reviews-per-10k stress grid;
- PaySim is synthetic and its steps are not production SLA units.
