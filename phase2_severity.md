# Phase 2: Drift Severity

## Severity Rule
A drift score is classified as LOW if it falls within the no-drift baseline range
(10 random 50/50 splits of NFCorpus, same sample size n=150), MEDIUM if it is up to
5x the baseline maximum, and HIGH if it exceeds 5x the baseline maximum.

## Results

| Comparison | Baseline Range (MMD) | Real Score (MMD) | Severity |
|---|---|---|---|
| Documents | 0.0000392 – 0.0000808 | 0.00102 | HIGH |
| Queries | 0.0000511 – 0.0000668 | 0.00092 | HIGH |
| Pooled | 0.0000392 – 0.0000808 | 0.00069 | HIGH |

## Interpretation
All three comparisons — documents, queries, and pooled — land well above the
no-drift baseline, roughly 9x to 14x the maximum score produced by random
splits of the same corpus. This confirms that the NFCorpus vs SciFact shift
is not within the range of ordinary sampling variation; it represents a
genuine, severe distributional change. Query drift is slightly higher than
document drift, suggesting the way questions are phrased differs even more
sharply between domains than the underlying document content itself.