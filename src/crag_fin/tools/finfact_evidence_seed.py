"""Seed a ChromaDB collection from Fin-Fact's own evidence sentences.

The Day-10 plan's `crag_fin_news_v0` index produced a degenerate Macro-F1
(both modes tied at 0.133) because RSS-pulled 2026 UK/EU financial news
does not overlap with Fin-Fact's 2023 US-political claims. The real fix
is corpus alignment: Fin-Fact ships its own `evidence` field per entry,
a list of `{sentence, hrefs}` dicts pulled by the fact-checker from the
sources they consulted. By design these are topically aligned with the
gold label.

This module ingests every evidence sentence into a fresh ChromaDB
collection (`finfact_evidence_v0`), tagged with the source href (so the
credibility prior still works) and the gold label of the parent entry
(so we can later audit retrieval leakage / data hygiene).

Run via:
    python -m crag_fin.tools.finfact_evidence_seed
or with overrides:
    python -m crag_fin.tools.finfact_evidence_seed --limit 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_MIN = 40  # drop very short evidence snippets
EMBED_BATCH = 64


@dataclass(frozen=True)
class EvidenceRow:
    entry_idx: int
    claim: str
    gold: str
    sentence: str
    href: str  # primary URL (first href in the list) or "" if empty


def _iter_evidence(finfact: list[dict], limit: int | None = None) -> list[EvidenceRow]:
    rows: list[EvidenceRow] = []
    for i, entry in enumerate(finfact):
        if limit is not None and i >= limit:
            break
        evidence = entry.get("evidence") or []
        if not isinstance(evidence, list):
            continue
        claim = entry.get("claim", "")
        gold = entry.get("label", "NEI")
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            sentence = str(ev.get("sentence", "")).strip()
            if len(sentence) < CHUNK_MIN:
                continue
            hrefs = ev.get("hrefs") or []
            href = str(hrefs[0]) if isinstance(hrefs, list) and hrefs else ""
            rows.append(
                EvidenceRow(
                    entry_idx=i,
                    claim=claim,
                    gold=gold,
                    sentence=sentence,
                    href=href,
                )
            )
    return rows


def seed(
    finfact_path: Path,
    chroma_dir: Path,
    collection_name: str,
    limit: int | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Build the finfact_evidence collection. Returns a manifest dict."""
    finfact_path = Path(finfact_path)
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    finfact = json.loads(finfact_path.read_text())
    rows = _iter_evidence(finfact, limit=limit)
    print(f"[seed] {len(rows)} evidence sentences from {finfact_path.name}")

    embedder = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Fresh start each run, reproducible by design.
    try:
        client.delete_collection(collection_name)
    except (ValueError, Exception):  # noqa: BLE001 — chromadb is inconsistent on missing
        pass
    collection = client.create_collection(name=collection_name)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for offset, r in enumerate(rows):
        # Source carries the href so credibility_prior.credibility() works;
        # falls back to a synthetic finfact:// URI if no href was provided.
        source = r.href or f"finfact://entry/{r.entry_idx}"
        # Full SHA-1 + offset suffix — chromadb requires globally unique ids,
        # and Fin-Fact has many near-duplicate evidence sentences across
        # entries. Full hash keeps it stable; the offset disambiguates the
        # remaining exact-duplicate cases.
        base = hashlib.sha1(f"{r.entry_idx}::{r.sentence}".encode()).hexdigest()
        doc_id = base if base not in seen_ids else f"{base}-{offset}"
        seen_ids.add(doc_id)
        documents.append(r.sentence)
        metadatas.append(
            {
                "source": source,
                "claim": r.claim[:300],
                "gold": r.gold,
                "entry_idx": r.entry_idx,
                "section": "evidence",
            }
        )
        ids.append(doc_id)

    # Batch the writes so we don't blow chromadb's per-call size limit.
    for start in range(0, len(documents), EMBED_BATCH):
        end = start + EMBED_BATCH
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]
        embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,  # type: ignore[arg-type]
            ids=batch_ids,
            embeddings=embeddings,
        )

    final = collection.count()
    manifest = {
        "finfact_path": str(finfact_path),
        "collection_name": collection_name,
        "rows_ingested": final,
        "limit_entries": limit,
        "embedding_model": embedding_model,
    }
    print(f"[seed] done: {final} chunks in {collection_name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finfact",
        default="data/finfact/raw/finfact.json",
    )
    parser.add_argument("--chroma-dir", default=os.environ["FINFACT_CHROMA_DIR"])
    parser.add_argument(
        "--collection",
        default=os.environ.get("FINFACT_COLLECTION", "finfact_evidence_v0"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest evidence from the first N Fin-Fact entries (debug).",
    )
    args = parser.parse_args()

    seed(
        finfact_path=Path(args.finfact),
        chroma_dir=Path(args.chroma_dir),
        collection_name=args.collection,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
