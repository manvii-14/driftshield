
"""DriftShield Phase 2 — Person 3: Embedding fingerprints."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


ROOT = Path(__file__).resolve().parent
K = 12
RANDOM_STATE = 0


def load_ids(filename):
    with open(ROOT / filename, encoding="utf-8") as f:
        ids = json.load(f)
    return [str(x) for x in ids]


def load_texts(dataset_name, ids):
    """Return texts in precisely the saved embedding-ID order."""
    corpus = load_dataset(
        dataset_name, "corpus", split="corpus"
    )

    text_by_id = {
        str(row["_id"]): (
            str(row.get("title", "")) + " " +
            str(row.get("text", ""))
        ).strip()
        for row in corpus
    }

    missing = [doc_id for doc_id in ids if doc_id not in text_by_id]
    if missing:
        raise ValueError(
            f"{dataset_name}: {len(missing)} IDs are missing: "
            f"{missing[:10]}"
        )

    return [text_by_id[doc_id] for doc_id in ids]


def validate_embeddings(X, name):
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def main():
    print("Loading saved embeddings and IDs...")

    before = np.load(ROOT / "emb_before.npy")
    after = np.load(ROOT / "emb_after.npy")
    ids_before = load_ids("ids_before.json")
    ids_after = load_ids("ids_after.json")

    validate_embeddings(before, "NFCorpus")
    validate_embeddings(after, "SciFact")

    if before.shape[1] != after.shape[1]:
        raise ValueError("Embedding dimensions do not match.")
    if len(ids_before) != len(before):
        raise ValueError("NFCorpus ID count does not match embeddings.")
    if len(ids_after) != len(after):
        raise ValueError("SciFact ID count does not match embeddings.")
    if len(set(ids_before)) != len(ids_before):
        raise ValueError("Duplicate NFCorpus IDs.")
    if len(set(ids_after)) != len(ids_after):
        raise ValueError("Duplicate SciFact IDs.")

    print("Loading original document texts...")
    texts_before = load_texts("BeIR/nfcorpus", ids_before)
    texts_after = load_texts("BeIR/scifact", ids_after)

    X = np.vstack([before, after])
    texts = texts_before + texts_after
    domains = np.array(
        ["NFCorpus"] * len(before) + ["SciFact"] * len(after)
    )
    ids = ids_before + ids_after

    if len(texts) != len(X) or len(domains) != len(X):
        raise ValueError("Combined data alignment failed.")

    print(f"NFCorpus: {len(before)} documents")
    print(f"SciFact: {len(after)} documents")
    print(f"Embedding dimension: {X.shape[1]}")

    # PCA visualization.
    print("Generating PCA embedding map...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    for domain in ["NFCorpus", "SciFact"]:
        mask = domains == domain
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.45, label=domain
        )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )
    ax.set_title("DriftShield: NFCorpus vs SciFact Embedding Map")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "embedding_map.png", dpi=200)
    plt.close(fig)

    # KMeans clustering.
    print("Clustering combined embeddings...")
    if len(X) < K:
        raise ValueError(f"Need at least {K} documents for KMeans.")

    model = KMeans(
        n_clusters=K, random_state=RANDOM_STATE, n_init=10
    )
    labels = model.fit_predict(X)

    # TF-IDF terms per cluster, computed on the original texts.
    print("Extracting cluster terms...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())
    except ValueError as exc:
        raise ValueError(
            "TF-IDF could not create a vocabulary from the texts."
        ) from exc

    rows = []
    cluster_terms = {}

    n_before = len(before)
    n_after = len(after)

    for cluster_id in range(K):
        mask = labels == cluster_id
        before_mask = mask & (domains == "NFCorpus")
        after_mask = mask & (domains == "SciFact")

        total = int(mask.sum())
        before_count = int(before_mask.sum())
        after_count = int(after_mask.sum())

        # Rank terms by their mean TF-IDF weight within the cluster.
        mean_scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_indices = np.argsort(mean_scores)[::-1]
        top_terms = [
            str(terms[i]) for i in top_indices
            if mean_scores[i] > 0
        ][:10]
        cluster_terms[cluster_id] = top_terms

        before_within = 100 * before_count / total if total else 0
        after_within = 100 * after_count / total if total else 0
        before_domain_share = 100 * before_count / n_before
        after_domain_share = 100 * after_count / n_after

        rows.append({
            "cluster_id": cluster_id,
            "total_documents": total,
            "nfcorpus_count": before_count,
            "scifact_count": after_count,
            "nfcorpus_within_cluster_pct": before_within,
            "scifact_within_cluster_pct": after_within,
            "nfcorpus_domain_share_pct": before_domain_share,
            "scifact_domain_share_pct": after_domain_share,
            "domain_share_difference_percentage_points":
                after_domain_share - before_domain_share,
            "top_terms": ", ".join(top_terms),
        })

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "cluster_fingerprints.csv", index=False)

    # Grouped bar chart: each value is a share of that domain's documents.
    print("Generating cluster-share chart...")
    positions = np.arange(K)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        positions - width / 2,
        df["nfcorpus_domain_share_pct"],
        width, label="NFCorpus"
    )
    ax.bar(
        positions + width / 2,
        df["scifact_domain_share_pct"],
        width, label="SciFact"
    )
    ax.set_xlabel("KMeans cluster")
    ax.set_ylabel("Share of domain documents (%)")
    ax.set_title("Document Distribution Across Clusters")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in range(K)])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "cluster_shares.png", dpi=200)
    plt.close(fig)

    # Markdown report generated from the actual results.
    report = [
        "# DriftShield Phase 2: Fingerprint Analysis",
        "",
        "## Objective",
        "",
        "Characterize differences between NFCorpus (before/reference) "
        "and SciFact (after/comparison) using saved document embeddings.",
        "",
        "## Dataset and methodology",
        "",
        f"- NFCorpus documents: {n_before:,}",
        f"- SciFact documents: {n_after:,}",
        f"- Embedding dimensions: {X.shape[1]}",
        f"- Total clustered documents: {len(X):,}",
        f"- Embedding model used by the existing pipeline: "
        "`all-MiniLM-L6-v2`.",
        "- Document text is aligned to embeddings using saved document IDs.",
        "- PCA with two components is used for visualization.",
        "- KMeans uses 12 clusters, random_state=0, and n_init=10.",
        "- Top terms are ranked by mean TF-IDF within each cluster.",
        "",
        "## Embedding map",
        "",
        "![PCA embedding map](embedding_map.png)",
        "",
        "## Cluster shares",
        "",
        "![Cluster shares](cluster_shares.png)",
        "",
        "## Cluster results",
        "",
        "| Cluster | Total | NFCorpus | SciFact | NFCorpus within cluster (%) | SciFact within cluster (%) | NFCorpus domain share (%) | SciFact domain share (%) | Difference (pp) | Top terms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        report.append(
            f"| {row['cluster_id']} | {row['total_documents']} "
            f"| {row['nfcorpus_count']} | {row['scifact_count']} "
            f"| {row['nfcorpus_within_cluster_pct']:.2f} "
            f"| {row['scifact_within_cluster_pct']:.2f} "
            f"| {row['nfcorpus_domain_share_pct']:.2f} "
            f"| {row['scifact_domain_share_pct']:.2f} "
            f"| {row['domain_share_difference_percentage_points']:.2f} "
            f"| {row['top_terms']} |"
        )

    largest = sorted(
        rows,
        key=lambda r: abs(
            r["domain_share_difference_percentage_points"]
        ),
        reverse=True
    )[:3]

    report += [
        "",
        "## Interpretation",
        "",
        "Clusters with positive domain-share differences are relatively "
        "more represented in SciFact; negative differences indicate "
        "relatively greater NFCorpus representation.",
        "",
    ]

    for row in largest:
        direction = (
            "SciFact" if
            row["domain_share_difference_percentage_points"] > 0
            else "NFCorpus"
        )
        report.append(
            f"- Cluster {row['cluster_id']} has a domain-share difference "
            f"of {row['domain_share_difference_percentage_points']:.2f} "
            f"percentage points toward {direction}. Representative terms: "
            f"{', '.join(cluster_terms[row['cluster_id']]) or 'none'}."
        )

    report += [
        "",
        "These clusters describe differences in embedding-space "
        "representation. They do not establish that domain drift caused "
        "a change in retrieval performance.",
        "",
        "## Reproducibility",
        "",
        "Run `python fingerprints.py` from the repository directory. "
        "Required packages include numpy, pandas, matplotlib, scikit-learn, "
        "and datasets.",
        "",
        "The CSV contains the complete per-cluster statistics.",
    ]

    (ROOT / "phase2_fingerprints.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    # Sanity checks.
    assert len(df) == K
    assert int(df["total_documents"].sum()) == len(X)
    assert int(df["nfcorpus_count"].sum()) == n_before
    assert int(df["scifact_count"].sum()) == n_after
    assert np.allclose(
        df["nfcorpus_domain_share_pct"].sum(), 100.0
    )
    assert np.allclose(
        df["scifact_domain_share_pct"].sum(), 100.0
    )

    print("\nAll validation checks passed.")
    print("Created:")
    for name in [
        "fingerprints.py",
        "embedding_map.png",
        "cluster_shares.png",
        "cluster_fingerprints.csv",
        "phase2_fingerprints.md",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
