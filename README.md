# DriftShield
## Statistical Embedding-Drift Monitoring for RAG Systems

DriftShield is a research prototype for examining embedding distribution
changes between a reference retrieval domain and a comparison domain.
This phase compares NFCorpus (reference/before) with SciFact
(comparison/after).

## Datasets

- **NFCorpus:** biomedical information retrieval dataset.
- **SciFact:** scientific claim verification dataset used as the
  comparison domain.
- The repository contains saved document and query embeddings and
  their corresponding ID lists. The analysis scripts use these artifacts
  rather than requiring document re-embedding.

Dataset source: https://huggingface.co/datasets/BeIR/nfcorpus
SciFact source: https://huggingface.co/datasets/BeIR/scifact

## Phase 2 analyses

1. **Drift types:** cosine-centroid comparison, Maximum Mean Discrepancy
   (MMD), and Energy Distance.
2. **Drift severity:** compares real-domain scores with a randomly split
   no-drift baseline.
3. **Drift fingerprints:** PCA visualization, KMeans clustering, and
   TF-IDF terms to describe cluster-level domain representation.

Read [phase2_summary.md](phase2_summary.md) for the combined report.

## Environment setup

Use Python 3.10 or newer where compatible with the installed dependencies.

Create and activate a virtual environment if desired:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install numpy scipy scikit-learn matplotlib pandas datasets faiss-cpu sentence-transformers
```

## Run the analyses

Run these commands from the repository root.

```bash
python phase2_types.py
python severity.py
python fingerprints.py
```

The fingerprint script uses the saved `emb_before.npy` and
`emb_after.npy` embeddings and the corresponding document ID files.
It produces:

- `embedding_map.png`
- `cluster_shares.png`
- `cluster_fingerprints.csv`
- `phase2_fingerprints.md`

The drift-type and severity scripts produce their respective tables,
reports, and charts. Consult each script and report for its exact outputs.

## Important interpretation note

Embedding drift indicates a distribution difference, not necessarily
a loss in retrieval quality. Retrieval metrics such as Hit Rate@K,
Recall@K, and MRR should be evaluated separately. Cluster terms and
PCA plots are descriptive and do not establish causality.

## Reproducibility

The repository includes saved embeddings, document IDs, analysis
scripts, and reports. Dataset access may be required when retrieving
original document text. Exact results depend on the saved artifacts,
dependency versions, random seeds, and script configuration.
