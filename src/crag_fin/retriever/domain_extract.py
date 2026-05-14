"""Robust publisher-domain extraction.

The Day-2 `credibility_prior.extract_domain` is `urlparse`-only and fails
on regional subdomains, AMP pages, URL shorteners, and bare-domain inputs
that contain a path. Day 9 replaces it with a `tldextract`-based version
that:

  - collapses subdomains (`uk.reuters.com` -> `reuters.com`)
  - handles AMP pages (`www.bbc.co.uk/amp/...` -> `bbc.co.uk`)
  - returns None for URL shorteners (`bit.ly`, `t.co`, `lnkd.in`) so the
    caller knows to follow the redirect first
  - tolerates missing scheme (`reuters.com/x/y`) and case
  - extracts the first URL out of mixed text via regex when no obvious
    URL or domain is present

The naive Day-2 `extract_domain` stays callable from `credibility_prior`
as a "naive baseline" so we can A/B in evaluations.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import tldextract

_URL_RE = re.compile(r"https?://[^\s\]\)\>]+", re.IGNORECASE)
_SHORTENERS = frozenset(
    {
        "bit.ly",
        "t.co",
        "lnkd.in",
        "buff.ly",
        "ow.ly",
        "tinyurl.com",
        "goo.gl",
        "is.gd",
        "rebrand.ly",
        "shorturl.at",
    }
)
_AMP_PATH_PREFIXES = ("amp/", "amp.", "/amp/")


def _strip_amp(host: str) -> str:
    """Drop a leading 'amp.' subdomain if present."""
    if host.startswith("amp."):
        return host[4:]
    return host


def extract_domain(url_or_text: str) -> str | None:
    """Return a normalised second-level + TLD domain, or None.

    Returns None for: empty input, URL shorteners (the caller must follow
    redirect first), inputs with no parseable host.
    """
    if not url_or_text:
        return None
    raw = url_or_text.strip()

    # If the input is free text, pull the first URL out of it.
    match = _URL_RE.search(raw)
    candidate = match.group(0) if match else raw

    # Make sure tldextract sees a scheme so it parses netloc reliably.
    if "://" not in candidate:
        candidate = f"//{candidate}"

    parsed = urlparse(candidate, scheme="")
    host = (parsed.netloc or parsed.path).lower().split("/", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    host = _strip_amp(host)
    if not host:
        return None

    if host in _SHORTENERS:
        return None

    extracted = tldextract.extract(host)
    if not extracted.domain or not extracted.suffix:
        return None
    return f"{extracted.domain}.{extracted.suffix}"


def is_shortener(url_or_domain: str) -> bool:
    """True if the input is a known URL-shortener domain."""
    if not url_or_domain:
        return False
    raw = url_or_domain.strip().lower()
    if "://" in raw:
        host = urlparse(raw).netloc
    else:
        host = raw.split("/", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host in _SHORTENERS
