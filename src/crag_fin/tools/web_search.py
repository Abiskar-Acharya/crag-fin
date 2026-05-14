"""Tavily web-search wrapper.

Returns a flat list of WebHit dataclasses so the reasoner / pipeline can
treat web results the same way it treats ChromaDB passages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    snippet: str
    source_domain: str


def _domain_from(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def tavily_search(query: str, max_results: int = 5) -> list[WebHit]:
    """Run a Tavily search. Returns [] if no API key is configured.

    Reads `TAVILY_API_KEY` from the environment. We import tavily lazily
    so unit tests that mock the call don't need the package available.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    from tavily import TavilyClient  # type: ignore[import-not-found]

    client = TavilyClient(api_key=api_key)
    raw = client.search(query=query, max_results=max_results)
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [
        WebHit(
            title=str(r.get("title", "")),
            url=str(r.get("url", "")),
            snippet=str(r.get("content", "")),
            source_domain=_domain_from(str(r.get("url", ""))),
        )
        for r in results[:max_results]
    ]
