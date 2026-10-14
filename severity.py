import numpy as np
from driftlib import drift_report
import matplotlib.pyplot as plt

# ---- Step 1: Load the four files ----
emb_before = np.load("emb_before.npy")   # NFCorpus documents
emb_after  = np.load("emb_after.npy")    # SciFact documents
q_before   = np.load("q_before.npy")     # NFCorpus queries
q_after    = np.load("q_after.npy")      # SciFact queries

print("Shapes:", emb_before.shape, emb_after.shape, q_before.shape, q_after.shape)

N = 150  # sample size, used consistently everywhere

KEY_MMD = "mmd"
KEY_COSINE = "cosine_centroid"
KEY_ENERGY = "energy"

# ---- Step 2: Build the "no drift" baseline (10 random splits) ----
def build_baseline(data, n_runs=10, sample_size=N):
    results = {"mmd": [], "cosine": [], "energy": []}
    for seed in range(n_runs):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(data))
        half = len(data) // 2
        half1 = data[idx[:half]]
        half2 = data[idx[half:half * 2]]
        report = drift_report(half1, half2, n=sample_size)
        results["mmd"].append(report[KEY_MMD])
        results["cosine"].append(report[KEY_COSINE])
        results["energy"].append(report[KEY_ENERGY])
    return results

print("Building document baseline (10 runs)...")
doc_baseline = build_baseline(emb_before)

print("Building query baseline (10 runs)...")
query_baseline = build_baseline(q_before)

print("Doc baseline MMD range:", min(doc_baseline["mmd"]), "to", max(doc_baseline["mmd"]))
print("Query baseline MMD range:", min(query_baseline["mmd"]), "to", max(query_baseline["mmd"]))

# ---- Step 3: Real NFCorpus vs SciFact scores (random samples) ----
rng_real = np.random.RandomState(123)

def pick(X, n=N):
    return X[rng_real.choice(len(X), min(n, len(X)), replace=False)]

print("Computing real document drift...")
doc_report = drift_report(pick(emb_before), pick(emb_after), n=N)

print("Computing real query drift...")
query_report = drift_report(pick(q_before), pick(q_after), n=N)

print("Computing pooled drift (docs + queries combined)...")
pooled_before = np.vstack([pick(emb_before, N // 2), pick(q_before, N // 2)])
pooled_after  = np.vstack([pick(emb_after, N // 2), pick(q_after, N // 2)])
pooled_report = drift_report(pooled_before, pooled_after, n=N)

print("Doc MMD:", doc_report[KEY_MMD])
print("Query MMD:", query_report[KEY_MMD])
print("Pooled MMD:", pooled_report[KEY_MMD])

# ---- Step 4: Define severity and classify ----
def classify_severity(score, baseline_scores):
    baseline_max = max(baseline_scores)
    if score <= baseline_max:
        return "LOW"
    elif score <= baseline_max * 5:
        return "MEDIUM"
    else:
        return "HIGH"

doc_severity    = classify_severity(doc_report[KEY_MMD], doc_baseline["mmd"])
query_severity  = classify_severity(query_report[KEY_MMD], query_baseline["mmd"])
pooled_severity = classify_severity(pooled_report[KEY_MMD], doc_baseline["mmd"])

print("Document drift severity:", doc_severity)
print("Query drift severity:", query_severity)
print("Pooled drift severity:", pooled_severity)

# Ratio of each real score to its baseline maximum (for the report text)
print("Doc ratio to baseline max: %.1fx" % (doc_report[KEY_MMD] / max(doc_baseline["mmd"])))
print("Query ratio to baseline max: %.1fx" % (query_report[KEY_MMD] / max(query_baseline["mmd"])))
print("Pooled ratio to baseline max: %.1fx" % (pooled_report[KEY_MMD] / max(doc_baseline["mmd"])))

# ---- Step 5: Chart ----
fig, ax = plt.subplots(figsize=(8, 5))

baseline_min = min(doc_baseline["mmd"])
baseline_max = max(doc_baseline["mmd"])
ax.axhspan(baseline_min, baseline_max, color="lightgray", alpha=0.6, label="No-drift baseline range")

labels = ["Documents", "Queries", "Pooled"]
scores = [doc_report[KEY_MMD], query_report[KEY_MMD], pooled_report[KEY_MMD]]
severities = [doc_severity, query_severity, pooled_severity]
colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
point_colors = [colors[s] for s in severities]

ax.scatter(labels, scores, color=point_colors, s=150, zorder=5)
for i, (label, score, sev) in enumerate(zip(labels, scores, severities)):
    ax.annotate(sev, (i, score), textcoords="offset points", xytext=(0, 10), ha="center")

ax.set_ylabel("MMD score")
ax.set_title("Drift Severity: NFCorpus vs SciFact")
ax.legend()
plt.savefig("severity_chart.png")
print("Saved severity_chart.png")