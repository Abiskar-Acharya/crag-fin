"""Seed the financial-news ChromaDB collection from a curated RSS feed list.

Day 2 minimum viable scope: pull entries from each feed in `configs/rss_feeds.yaml`,
use the RSS-provided summary (rather than fetching full HTML) to keep latency low,
chunk by paragraph into ~500-char chunks, embed with the same SentenceTransformer
ArXivMind uses (`all-MiniLM-L6-v2`), and write to a fresh ChromaDB collection at
`NEWS_CHROMA_DIR / NEWS_COLLECTION`. The `source` metadata carries the article URL
so the Day-2 credibility prior can look up the publisher domain.

Run via:
    python -m crag_fin.tools.news_seed
or with overrides:
    python -m crag_fin.tools.news_seed --max-per-feed 25 --feeds configs/rss_feeds.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import feedparser
import yaml
from sentence_transformers import SentenceTransformer

CHUNK_TARGET = 500  # characters
CHUNK_MIN = 80  # drop chunks shorter than this (likely cruft)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FeedSpec:
    name: str
    url: str
    tier: int
    domain: str


@dataclass(frozen=True)
class Article:
    title: str
    url: str
    domain: str
    text: str


def load_feed_specs(yaml_path: Path) -> list[FeedSpec]:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return [
        FeedSpec(
            name=item["name"],
            url=item["url"],
            tier=int(item["tier"]),
            domain=item["domain"],
        )
        for item in raw["feeds"]
    ]


def _clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    no_tags = _TAG_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", no_tags).strip()


def fetch_articles(spec: FeedSpec, max_entries: int = 20) -> list[Article]:
    """Pull up to `max_entries` articles from one RSS feed.

    Uses the RSS-provided summary as the body — full-article fetch is
    deferred to a later phase to keep Day-2 wall time bounded.
    """
    parsed = feedparser.parse(spec.url)
    if parsed.bozo:
        # Network or parse error; log + skip rather than crashing the whole seed.
        err = parsed.get("bozo_exception", "parse failed")
        print(f"  [warn] {spec.name}: {err}", file=sys.stderr)
        return []

    out: list[Article] = []
    for entry in parsed.entries[:max_entries]:
        title = _clean(entry.get("title", ""))
        url = entry.get("link", "")
        summary = _clean(entry.get("summary", "") or entry.get("description", ""))
        body = f"{title}. {summary}".strip()
        if len(body) < CHUNK_MIN:
            continue
        out.append(Article(title=title, url=url, domain=spec.domain, text=body))
    return out


def chunk_text(text: str, target: int = CHUNK_TARGET) -> list[str]:
    """Split text into ~target-character chunks at paragraph or sentence boundaries.

    Simple and deterministic. News bodies are usually short enough that
    a single article produces 1-3 chunks.
    """
    if len(text) <= target:
        return [text]
    chunks: list[str] = []
    paragraphs = re.split(r"(?<=[.!?])\s+", text)
    buf = ""
    for sent in paragraphs:
        if not sent.strip():
            continue
        if len(buf) + len(sent) + 1 <= target:
            buf = f"{buf} {sent}".strip()
        else:
            if len(buf) >= CHUNK_MIN:
                chunks.append(buf)
            buf = sent
    if len(buf) >= CHUNK_MIN:
        chunks.append(buf)
    return chunks


def seed(
    feeds_yaml: Path,
    chroma_dir: Path,
    collection_name: str,
    max_per_feed: int = 20,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """Pull articles from every feed, embed, and write to a fresh collection.

    Returns a manifest dict {feeds, articles, chunks, collection_count}.
    """
    feeds_yaml = Path(feeds_yaml)
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    specs = load_feed_specs(feeds_yaml)
    print(f"[seed] loading {len(specs)} feeds from {feeds_yaml}")

    embedder = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Start fresh each run so the Day-2 demo is reproducible.
    try:
        client.delete_collection(collection_name)
    except (ValueError, Exception):  # noqa: BLE001 — chromadb signals "not found" inconsistently
        pass
    collection = client.create_collection(name=collection_name)

    total_articles = 0
    total_chunks = 0
    for spec in specs:
        articles = fetch_articles(spec, max_entries=max_per_feed)
        print(f"  [{spec.name}] {len(articles)} articles")
        total_articles += len(articles)

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for art in articles:
            for i, chunk in enumerate(chunk_text(art.text)):
                doc_id = hashlib.sha1(f"{art.url}::{i}".encode()).hexdigest()[:16]
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": art.url or spec.domain,
                        "domain": art.domain,
                        "tier": spec.tier,
                        "title": art.title[:200],
                        "section": f"chunk_{i}",
                    }
                )
                ids.append(doc_id)

        if not documents:
            continue
        embeddings = embedder.encode(documents, show_progress_bar=False).tolist()
        collection.add(
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
            embeddings=embeddings,
        )
        total_chunks += len(documents)

    final_count = collection.count()
    manifest = {
        "feeds": len(specs),
        "articles": total_articles,
        "chunks": total_chunks,
        "collection_count": final_count,
        "collection_name": collection_name,
    }
    print(f"[seed] done: {total_articles} articles -> {final_count} chunks in {collection_name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feeds",
        default=os.environ.get("RSS_FEEDS_CONFIG", "configs/rss_feeds.yaml"),
    )
    parser.add_argument("--chroma-dir", default=os.environ["NEWS_CHROMA_DIR"])
    parser.add_argument(
        "--collection",
        default=os.environ.get("NEWS_COLLECTION", "crag_fin_news_v0"),
    )
    parser.add_argument("--max-per-feed", type=int, default=20)
    args = parser.parse_args()

    seed(
        feeds_yaml=Path(args.feeds),
        chroma_dir=Path(args.chroma_dir),
        collection_name=args.collection,
        max_per_feed=args.max_per_feed,
    )


if __name__ == "__main__":
    main()
