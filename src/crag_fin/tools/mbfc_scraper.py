"""MBFC (Media Bias / Fact Check) tier scraper.

MBFC publishes outlet bias + factual-reporting ratings. We scrape per-outlet
pages, rate-limited at 1 req/sec, caching HTML to a local dir so reruns are
cheap. Output is a CSV with columns:
    [domain, mbfc_factual_reporting, mbfc_bias_label, mbfc_url]

The CSV is gitignored (data/source_credibility/mbfc_*). Day 8 of the plan
uses the result; Day 9 hardens domain extraction against the entries.

Note: MBFC's page layout changes occasionally. The selectors below are
written conservatively (look for known labels rather than tight CSS paths)
and the scraper logs + skips outlets it can't parse, rather than crashing
the whole run.
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DEFAULT_RATE_LIMIT_SEC = 1.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/537.36 "
    "crag-fin/0.1 (research) +https://github.com/Abiskar-Acharya"
)


@dataclass(frozen=True)
class MBFCEntry:
    domain: str
    factual_reporting: str  # VERY_HIGH / HIGH / MOSTLY_FACTUAL / MIXED / LOW / VERY_LOW
    bias_label: str  # LEFT / LEFT_CENTER / LEAST_BIASED / RIGHT_CENTER / RIGHT / ...
    mbfc_url: str


# Stop the label as soon as we hit lowercase, digits, or the next known
# section marker. MBFC labels are single tokens or hyphen/space-joined
# uppercase words (e.g. "VERY HIGH", "LEFT-CENTER", "MOSTLY FACTUAL").
# Case-sensitive: MBFC uses these exact section labels and uppercase tier
# values. The non-greedy class stops at the next known marker or any
# lowercase letter, so "HIGH Source: ..." captures just "HIGH".
_LABEL_END = r"(?=\s+(?:Source|Bias|Factual|Country|Notes|Founded|Click|$)|\s*[a-z])"
_FACTUAL_RE = re.compile(r"Factual Reporting:\s*([A-Z][A-Z\- ]*?)" + _LABEL_END)
_BIAS_RE = re.compile(r"Bias Rating:\s*([A-Z][A-Z\- ]*?)" + _LABEL_END)
_DOMAIN_RE = re.compile(r"Source[s]?:\s*\[?\s*(https?://[^\s\]]+)")


def _normalise_label(label: str) -> str:
    return re.sub(r"\s+", "_", label.strip().upper())


def _extract_domain_from(url: str) -> str:
    from urllib.parse import urlparse

    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def parse_outlet_page(html: str, mbfc_url: str) -> MBFCEntry | None:
    """Parse one MBFC outlet page. Returns None on missing fields."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    factual_match = _FACTUAL_RE.search(text)
    bias_match = _BIAS_RE.search(text)
    domain_match = _DOMAIN_RE.search(text)
    if not (factual_match and bias_match and domain_match):
        return None

    return MBFCEntry(
        domain=_extract_domain_from(domain_match.group(1)),
        factual_reporting=_normalise_label(factual_match.group(1)),
        bias_label=_normalise_label(bias_match.group(1)),
        mbfc_url=mbfc_url,
    )


def fetch(url: str, cache_dir: Path, session: requests.Session) -> str:
    """Cached GET. Returns empty string on HTTP error (caller skips)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    cached = cache_dir / f"{key}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="ignore")
    resp = session.get(url, timeout=15, headers={"User-Agent": DEFAULT_USER_AGENT})
    if resp.status_code != 200:
        return ""
    text: str = resp.text
    cached.write_text(text, encoding="utf-8")
    return text


def scrape_outlets(
    outlet_urls: list[str],
    out_csv: Path,
    cache_dir: Path,
    rate_limit_sec: float = DEFAULT_RATE_LIMIT_SEC,
) -> int:
    """Scrape MBFC outlet pages, write CSV, return number of rows written.

    `outlet_urls` is the list of per-outlet MBFC URLs to scrape (e.g.
    https://mediabiasfactcheck.com/reuters/). Caller obtains this list from
    MBFC's category/index pages (kept out of this module to make the
    function easy to unit-test).
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    entries: list[MBFCEntry] = []
    for url in outlet_urls:
        html = fetch(url, cache_dir, session)
        if not html:
            continue
        entry = parse_outlet_page(html, url)
        if entry is not None:
            entries.append(entry)
        time.sleep(rate_limit_sec)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "mbfc_factual_reporting", "mbfc_bias_label", "mbfc_url"])
        for e in entries:
            writer.writerow([e.domain, e.factual_reporting, e.bias_label, e.mbfc_url])
    return len(entries)
