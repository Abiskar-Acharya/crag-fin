"""Hand-coded source-credibility prior.

A tier-based credibility score in [0, 1] is attached to a passage's publisher
domain so the retriever can rerank evidence in favour of more trustworthy
outlets. The tiers come from `concepts/concept_Source_Credibility.md` in the
project vault:

  Tier 1 (1.0) — wires + tier-1 broadsheet: Reuters, Bloomberg, FT, AP, WSJ
  Tier 2 (0.7) — mainstream broadsheet: BBC, Guardian, Telegraph
  Tier 3 (0.4) — topical / opinionated: Seeking Alpha, Motley Fool, regional blogs
  Tier 4 (0.1) — anonymous / social: Telegram, anonymous Twitter, Discord screenshots

Unknown domains receive `DEFAULT_CREDIBILITY = 0.5` (per the 30-day plan
Day-2 mitigation). Day 8 supersedes this hand-coded dict with the scraped MBFC
table; Day 9 replaces the naive `extract_domain` with a tldextract-based
implementation.
"""

from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_CREDIBILITY: float = 0.5

TIER_DICT: dict[str, float] = {
    # Tier 1 — 1.0
    "reuters.com": 1.0,
    "bloomberg.com": 1.0,
    "ft.com": 1.0,
    "apnews.com": 1.0,
    "wsj.com": 1.0,
    "nytimes.com": 1.0,
    "economist.com": 1.0,
    # Tier 2 — 0.7
    "bbc.co.uk": 0.7,
    "bbc.com": 0.7,
    "theguardian.com": 0.7,
    "guardian.co.uk": 0.7,
    "telegraph.co.uk": 0.7,
    "cnbc.com": 0.7,
    "cnn.com": 0.7,
    "forbes.com": 0.7,
    "businessinsider.com": 0.7,
    # Tier 3 — 0.4
    "seekingalpha.com": 0.4,
    "fool.com": 0.4,
    "marketwatch.com": 0.4,
    "benzinga.com": 0.4,
    "investorplace.com": 0.4,
    "zerohedge.com": 0.4,
    "yahoo.com": 0.4,
    "finance.yahoo.com": 0.4,
    # Tier 4 — 0.1
    "telegram.org": 0.1,
    "discord.com": 0.1,
    "reddit.com": 0.1,
    "4chan.org": 0.1,
}


def extract_domain(url_or_source: str) -> str:
    """Return a normalised domain string from a URL or bare publisher name.

    Day 2 implementation is intentionally naive: lowercase, strip `www.`,
    take the netloc if it looks like a URL, otherwise return the input
    lowercased. Day 9 replaces this with a tldextract-based version that
    handles subdomains (regional.bbc.co.uk -> bbc.co.uk) and AMP pages.
    """
    if not url_or_source:
        return ""
    raw = url_or_source.strip().lower()
    parsed = urlparse(raw if "://" in raw else f"//{raw}", scheme="")
    host = parsed.netloc or parsed.path
    host = host.split("/", 1)[0]  # drop trailing path if scheme was missing
    if host.startswith("www."):
        host = host[4:]
    return host


def credibility(source: str) -> float:
    """Map a passage's source field (URL or domain) to a credibility score.

    Returns DEFAULT_CREDIBILITY for unknown domains. Empty / falsy input
    also returns DEFAULT_CREDIBILITY rather than raising, so the retriever
    can stay branchless when applying the weight.
    """
    domain = extract_domain(source)
    if not domain:
        return DEFAULT_CREDIBILITY
    return TIER_DICT.get(domain, DEFAULT_CREDIBILITY)
