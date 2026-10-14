# Phase 2: Drift Severity

## Severity Rule
A drift score is classified as LOW if it falls within the no-drift baseline range
(10 random 50/50 splits of NFCorpus, same sample size n=150), MEDIUM if it is up to
5x the baseline maximum, and HIGH if it exceeds 5x the baseline maximum.

## Results

| Comparison | Baseline Range (MMD) | Real Score (MMD) | Ratio to baseline max | Severity |
|---|---|---|---|---|
| Documents | 0.0000392 – 0.0000809 | 0.000768 | 9.5x | HIGH |
| Queries | 0.0000511 – 0.0000665 | 0.000855 | 12.9x | HIGH |
| Pooled | 0.0000392 – 0.0000809 | 0.000813 | 10.0x | HIGH |

Each score is computed on a random sample of 150 per side. The baseline comes from 10 random 50/50 splits of NFCorpus.

## Interpretation
All three comparisons (documents, queries and pooled) land far above the no-drift baseline, between 9.5x and 12.9x the largest score produced by random splits of the same corpus. This confirms that the NFCorpus vs SciFact shift is outside ordinary sampling variation, so it counts as a severe distributional change. Query drift is slightly higher than document drift, but the two are of similar magnitude, so both the questions and the knowledge base shifted.