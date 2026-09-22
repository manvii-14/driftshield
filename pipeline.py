from datasets import load_dataset
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

print("Loading datasets...")
corpus_before = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
corpus_after = load_dataset("BeIR/scifact", "corpus")["corpus"]
print("Before docs:", len(corpus_before))
print("After docs:", len(corpus_after))

print("Loading embedding model (first time takes a minute)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Keep ids aligned with texts so retrieval can return ids, not raw text
ids_before = [doc["_id"] for doc in corpus_before]
texts_before = [doc["text"] for doc in corpus_before]

ids_after = [doc["_id"] for doc in corpus_after]
texts_after = [doc["text"] for doc in corpus_after]

print("Embedding before-corpus...")
emb_before = model.encode(texts_before, show_progress_bar=True)
print("Embedding after-corpus...")
emb_after = model.encode(texts_after, show_progress_bar=True)

np.save("emb_before.npy", emb_before)
np.save("emb_after.npy", emb_after)
print("Saved embeddings. Shapes:", emb_before.shape, emb_after.shape)

index = faiss.IndexFlatL2(emb_before.shape[1])
index.add(emb_before)

def retrieve(query, k=5):
    """Returns doc IDs — used for evaluation against qrels."""
    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, k)
    return [ids_before[i] for i in indices[0]]

def retrieve_text(query, k=5):
    """Returns doc text — used for human-readable sanity checks."""
    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, k)
    return [texts_before[i] for i in indices[0]]

# test it — print actual text for inspection
results = retrieve_text("What causes inflammation?", k=3)
for r in results:
    print(r[:200], "\n---")

print("Loading qrels and queries...")
qrels = load_dataset("BeIR/nfcorpus-qrels")["test"]
queries = load_dataset("BeIR/nfcorpus", "queries")["queries"]

# Pre-group qrels by query-id so we don't rescan the whole list per query
print("Indexing qrels...")
qrels_by_query = defaultdict(list)
for r in qrels:
    qrels_by_query[r["query-id"]].append(r["corpus-id"])

def hit_rate_at_k(queries, qrels_by_query, retrieve_fn, k=5):
    hits = 0
    total = 0
    for q in queries:
        retrieved = retrieve_fn(q["text"], k=k)
        relevant_ids = qrels_by_query.get(q["_id"], [])
        if relevant_ids:
            total += 1
            if any(doc in retrieved for doc in relevant_ids):
                hits += 1
    return hits / total if total else 0

print("Evaluating hit rate...")
score = hit_rate_at_k(queries, qrels_by_query, retrieve)
print("Hit rate:", score)