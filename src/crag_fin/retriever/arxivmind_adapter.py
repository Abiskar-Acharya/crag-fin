"""Adapter around the ArXivMind HybridRetriever.

Reuses the ArXivMind retrieval substrate (BM25 + dense + RRF + cross-encoder)
without modifying that repo. The adapter additionally supports:
- swapping the underlying ChromaDB collection at call time, so the same
  retriever can hit either the research-paper index or a financial-news
  collection seeded by `src/crag_fin/tools/news_seed.py`;
- credibility-weighted reranking, applied as a post-processing step on
  the dict list returned by `HybridRetriever.search`. Per Day-2 plan, we
  do NOT monkey-patch the underlying ArXivMind code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from crag_fin.retriever.credibility_prior import credibility as cred_lookup
from crag_fin.retriever.passage import Passage

_retriever_cache: dict[str, object] = {}


def _build_retriever(chroma_dir: str, collection_name: str):
    """Lazy-init a HybridRetriever for a (chroma_dir, collection) pair.

    Cached so the heavy cross-encoder model is only loaded once per pair.
    """
    cache_key = f"{chroma_dir}::{collection_name}"
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    repo = Path(os.environ["ARXIVMIND_REPO"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from app.retriever import HybridRetriever  # type: ignore[import-not-found]

    embed_model = os.environ.get("ARXIVMIND_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedder = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=collection_name)

    retriever = HybridRetriever(collection, embedder)
    if collection.count() > 0:
        retriever.build_bm25_index()

    _retriever_cache[cache_key] = retriever
    return retriever


def _get_default_retriever():
    chroma_dir = os.environ["ARXIVMIND_CHROMA_DIR"]
    collection_name = os.environ.get("ARXIVMIND_COLLECTION", "research_papers")
    return _build_retriever(chroma_dir, collection_name)


def _get_news_retriever():
    chroma_dir = os.environ["NEWS_CHROMA_DIR"]
    collection_name = os.environ.get("NEWS_COLLECTION", "crag_fin_news_v0")
    return _build_retriever(chroma_dir, collection_name)


def retrieve(
    query: str,
    k: int = 5,
    *,
    weighted: bool = False,
    collection: str | None = None,
) -> list[Passage]:
    """Retrieve top-k passages for a query.

    Args:
        query: the search query.
        k: number of passages to return.
        weighted: when True, multiply each candidate score by the publisher
            credibility (from `credibility_prior.credibility`) and re-sort
            before truncating to k. The original score is preserved on the
            returned Passage; the weighted score is set on `weighted_score`.
        collection: which named collection to query. Defaults to the
            ArXivMind research-papers collection; pass "news" to hit the
            Day-2 financial-news collection (`NEWS_COLLECTION`).

    Returns:
        list of Passage of length <= k. Empty when the underlying index has
        no documents.
    """
    if collection == "news":
        retriever = _get_news_retriever()
    else:
        retriever = _get_default_retriever()

    # Over-fetch when weighted so the post-rerank top-k still draws from a
    # rich candidate pool. Cap at 3k to bound latency.
    n_request = min(3 * k, 30) if weighted else k
    raw = retriever.search(query, n_results=n_request)

    # Day 2 weighting note: ArXivMind's cross-encoder normalises the top-1
    # rerank score to 1.0 (see glm-rag-pipeline/app/retriever.py:243-256),
    # which collapses dynamic range and makes naive `score * credibility`
    # ineffective at flipping rank-1. We use a rank-decayed credibility
    # weight instead:
    #   weighted_score = (credibility + 0.1) * RANK_DECAY ** rank
    # RANK_DECAY=0.7 means: a tier-1 source (cred=1.0) at rank 1 scores
    # 1.1 * 0.7 = 0.77, beating a tier-3 (cred=0.4) at rank 0 which scores
    # 0.5 * 1.0 = 0.5 — the intended swap. Day 10 eval revisits the
    # constants if uniform > weighted on Macro-F1.
    RANK_DECAY = 0.7
    CRED_FLOOR = 0.1  # keeps tier-4 (cred=0.1) from being zeroed out

    candidates: list[Passage] = []
    for rank, r in enumerate(raw):
        source = r.get("source", "unknown")
        base_score = 1.0 - float(r.get("distance", 0.0))
        cred = cred_lookup(source)
        weighted_score = (cred + CRED_FLOOR) * (RANK_DECAY ** rank) if weighted else None
        candidates.append(
            Passage(
                text=r.get("text", ""),
                source=source,
                score=base_score,
                section=r.get("section", ""),
                retrieval_method=r.get("retrieval_method", ""),
                credibility=cred,
                weighted_score=weighted_score,
            )
        )

    if weighted:
        candidates.sort(key=lambda p: p.weighted_score or 0.0, reverse=True)
    return candidates[:k]
