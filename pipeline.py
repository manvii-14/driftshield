import os
import json
import numpy as np
import faiss
from collections import defaultdict
from datasets import load_dataset

K = 5

print("Loading datasets...")
corpus_before = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
corpus_after = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries_before = load_dataset("BeIR/nfcorpus", "queries")["queries"]
queries_after = load_dataset("BeIR/scifact", "queries")["queries"]
qrels_before_raw = load_dataset("BeIR/nfcorpus-qrels")["test"]
qrels_after_raw = load_dataset("BeIR/scifact-qrels")["test"]

# Everything is cast to str: scifact uses integer ids in qrels, string ids elsewhere
ids_before = [str(d["_id"]) for d in corpus_before]
ids_after = [str(d["_id"]) for d in corpus_after]
texts_before = [d["text"] for d in corpus_before]
texts_after = [d["text"] for d in corpus_after]

def group_qrels(raw):
    grouped = defaultdict(set)
    for r in raw:
        grouped[str(r["query-id"])].add(str(r["corpus-id"]))
    return grouped

qrels_before = group_qrels(qrels_before_raw)
qrels_after = group_qrels(qrels_after_raw)

# Keep only queries that have test qrels (these are the ones we evaluate on)
qids_before = [str(q["_id"]) for q in queries_before if str(q["_id"]) in qrels_before]
qtexts_before = [q["text"] for q in queries_before if str(q["_id"]) in qrels_before]
qids_after = [str(q["_id"]) for q in queries_after if str(q["_id"]) in qrels_after]
qtexts_after = [q["text"] for q in queries_after if str(q["_id"]) in qrels_after]

print("Docs before/after:", len(ids_before), len(ids_after))
print("Queries before/after:", len(qids_before), len(qids_after))

# Load the embedding model only if something needs encoding
_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(texts, path):
    if os.path.exists(path):
        print("Loading cached", path)
        return np.load(path)
    print("Encoding ->", path)
    emb = get_model().encode(texts, show_progress_bar=True)
    np.save(path, emb)
    return emb

emb_before = embed(texts_before, "emb_before.npy")
emb_after = embed(texts_after, "emb_after.npy")
q_before = embed(qtexts_before, "q_before.npy")
q_after = embed(qtexts_after, "q_after.npy")
print("Shapes:", emb_before.shape, emb_after.shape, q_before.shape, q_after.shape)

# Save ids so teammates can line embeddings up with documents and queries
for name, obj in [("ids_before", ids_before), ("ids_after", ids_after),
                  ("qids_before", qids_before), ("qids_after", qids_after)]:
    with open(name + ".json", "w") as f:
        json.dump(obj, f)

def build_index(emb):
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    return index

def hit_rate(q_emb, qids, qrels, index, ids, k=K):
    _, idx = index.search(q_emb.astype("float32"), k)
    hits = 0
    for qid, row in zip(qids, idx):
        if qrels[qid] & {ids[i] for i in row}:
            hits += 1
    return hits / len(qids)

index_before = build_index(emb_before)
index_after = build_index(emb_after)

print("Hit rate (before):", hit_rate(q_before, qids_before, qrels_before, index_before, ids_before))
print("Hit rate (after, own queries):", hit_rate(q_after, qids_after, qrels_after, index_after, ids_after))