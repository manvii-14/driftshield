# Phase 2: Drift types

Sample size: 150 per side for every row, so scores are comparable.

| Pair | Cosine centroid | MMD | MMD p-value | Energy | Drift detected |
|---|---|---|---|---|---|
| Query drift (NFCorpus vs SciFact queries) | 0.3258 | 0.00091 | 0.0050 | 0.0618 | Yes |
| Document drift (NFCorpus vs SciFact docs) | 0.5341 | 0.00080 | 0.0050 | 0.0590 | Yes |
| Domain drift (docs + queries pooled) | 0.5313 | 0.00058 | 0.0050 | 0.0518 | Yes |
| Baseline: NFCorpus queries, half vs half | 0.9638 | 0.00005 | 0.8557 | 0.0246 | No |
| Baseline: NFCorpus docs, half vs half | 0.9979 | 0.00004 | 0.8458 | 0.0236 | No |

## Interpretation

All three drift types are detected by MMD, while same-domain baselines are not (p > 0.8), so the test separates real drift from sampling noise. Query drift (MMD 0.00091) and document drift (0.00080) are of similar magnitude, so both the questions and the knowledge base shifted. Cosine centroid and energy distance separate drift from baseline less cleanly than MMD, which supports using MMD as the primary test. Scores come from random samples of 150 per side, so they vary slightly between runs and between reports; the conclusions do not change.
