"""Hybrid retrieval: vector recall, then a lexical re-rank.

Vector search recalls `k * RECALL_FACTOR` candidates. Each is re-scored as

    score = vector_similarity + W_HEADING * heading_overlap + W_BODY * body_overlap

where the overlaps are the share of query terms found in the chunk's section heading
(its last, most specific part) and body. Help-centre docs put the task in the heading
("To Insert a Table of Contents"), so heading matches are a strong, cheap signal.
"""

from __future__ import annotations

from app.embeddings import Embedder, tokenize
from app.store import Hit, VectorStore

RECALL_FACTOR = 6
W_HEADING = 1.5
W_BODY = 0.2


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    q = set(tokenize(query))
    if not q:
        return hits
    rescored = []
    for h in hits:
        leaf = h.chunk.section.split(" > ")[-1]
        heading = set(tokenize(leaf))
        body = set(tokenize(h.chunk.text))
        score = h.score + W_HEADING * len(q & heading) / len(q) + W_BODY * len(q & body) / len(q)
        rescored.append(Hit(h.chunk, score))
    return sorted(rescored, key=lambda h: h.score, reverse=True)


def hybrid_search(
    store: VectorStore, embedder: Embedder, query: str, k: int = 5, product: str | None = None
) -> list[Hit]:
    vec = embedder.embed([query])[0]
    candidates = store.search(vec, k=k * RECALL_FACTOR, product=product)
    if not candidates and product:
        candidates = store.search(vec, k=k * RECALL_FACTOR)
    return rerank(query, candidates)[:k]
