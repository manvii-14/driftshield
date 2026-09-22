import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import energy_distance
import matplotlib.pyplot as plt

# Load embeddings
emb_before = np.load("emb_before.npy")
emb_after = np.load("emb_after.npy")
print("Loaded:", emb_before.shape, emb_after.shape)

# Subsample for speed
np.random.seed(42)
sample_before = emb_before[np.random.choice(len(emb_before), 500, replace=False)]
sample_after = emb_after[np.random.choice(len(emb_after), 500, replace=False)]

# --- Baseline: cosine centroid similarity ---
centroid_before = emb_before.mean(axis=0)
centroid_after = emb_after.mean(axis=0)
cosine_sim = np.dot(centroid_before, centroid_after) / (norm(centroid_before) * norm(centroid_after))
print("Cosine centroid similarity:", cosine_sim)

# --- MMD test (our own implementation, no TensorFlow needed) ---
def mmd_rbf(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]  # default heuristic
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_permutation_test(X, Y, n_permutations=100):
    observed = mmd_rbf(X, Y)
    combined = np.vstack([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_mmd = mmd_rbf(combined[:n], combined[n:])
        if perm_mmd >= observed:
            count += 1
    p_value = count / n_permutations
    return observed, p_value

print("Running MMD permutation test...")
mmd_score, p_value = mmd_permutation_test(sample_before, sample_after)
print(f"MMD score: {mmd_score:.5f} | p-value: {p_value:.4f} | Drift detected: {p_value < 0.05}")

# --- Energy distance test ---
dim_scores = [energy_distance(sample_before[:, i], sample_after[:, i]) for i in range(sample_before.shape[1])]
avg_energy = np.mean(dim_scores)
print("Average energy distance:", avg_energy)

# --- Final chart ---
hit_rate_before = 0.6253869969040248  # <-- fill in from Person 1's results.txt
hit_rate_after = 0.0   # <-- fill in from Person 1's results.txt

fig, ax1 = plt.subplots()
ax1.bar(["Before", "After"], [hit_rate_before, hit_rate_after], color="skyblue")
ax1.set_ylabel("Retrieval Hit-Rate")
plt.title("Drift Detected → RAG Quality Drop")
plt.savefig("result_chart.png")
print("Saved result_chart.png")