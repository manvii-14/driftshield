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

All three drift types are detected by MMD, while same-domain baselines are not (p > 0.8), so the test separates real drift from sampling noise. Query drift is the largest (MMD 0.00091), followed by document drift (0.00080), meaning the questions shifted more than the knowledge base. Pooling queries and documents gives the smallest score (0.00058), so drift should be measured per type rather than on pooled data. Cosine centroid and energy distance separate drift from baseline less cleanly than MMD, which supports using MMD as the primary test.
