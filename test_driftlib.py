import numpy as np
from driftlib import drift_report

emb_before = np.load("emb_before.npy")
emb_after = np.load("emb_after.npy")

print("NFCorpus vs SciFact (should show drift):")
print(drift_report(emb_before, emb_after))

# Sanity check: split NFCorpus in half, which should show NO drift
rng = np.random.default_rng(0)
perm = rng.permutation(len(emb_before))
half = len(perm) // 2
print("\nNFCorpus half vs half (should show no drift):")
print(drift_report(emb_before[perm[:half]], emb_before[perm[half:]]))