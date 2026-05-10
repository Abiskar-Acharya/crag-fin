"""Adapter around the ArXivMind HybridRetriever.

Reuses the ArXivMind retrieval substrate (BM25 + dense + RRF + cross-encoder)
without modifying that repo. On Day 1 the underlying ChromaDB holds research
papers, not financial news; the goal here is only to confirm the retrieval
*call* shape works end-to-end. Day 4 reseeds the index with financial passages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from crag_fin.retriever.passage import Passage

_retriever = None  # lazy singleton; cross-encoder load is heavy


def _get_retriever():
    """Lazy-init a HybridRetriever pointed at the ArXivMind chroma dir."""
    global _retriever
    if _retriever is not None:
        return _retriever

    repo = Path(os.environ["ARXIVMIND_REPO"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from app.retriever import HybridRetriever  # type: ignore[import-not-found]

    chroma_dir = Path(os.environ["ARXIVMIND_CHROMA_DIR"])
    collection_name = os.environ.get("ARXIVMIND_COLLECTION", "research_papers")
    embed_model = os.environ.get("ARXIVMIND_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    embedder = SentenceTransformer(embed_model)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)

    retriever = HybridRetriever(collection, embedder)
    if collection.count() > 0:
        retriever.build_bm25_index()

    _retriever = retriever
    return retriever


def retrieve(query: str, k: int = 5) -> list[Passage]:
    """Retrieve top-k passages for a query.

    Wraps ArXivMind's HybridRetriever.search and converts its dict output to
    the project's Passage dataclass. Returns [] if the underlying index is
    empty.
    """
    retriever = _get_retriever()
    raw = retriever.search(query, n_results=k)
    return [
        Passage(
            text=r.get("text", ""),
            source=r.get("source", "unknown"),
            # HybridRetriever returns "distance" in [0,1]; convert to a score
            # where higher = more relevant.
            score=1.0 - float(r.get("distance", 0.0)),
            section=r.get("section", ""),
            retrieval_method=r.get("retrieval_method", ""),
        )
        for r in raw
    ]
