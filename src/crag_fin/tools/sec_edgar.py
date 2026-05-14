"""SEC EDGAR lookup — recent filings by company name or ticker.

EDGAR's free API requires only a polite User-Agent header (set via
SEC_EDGAR_USER_AGENT). We hit two endpoints:
  1. company tickers JSON → CIK lookup
  2. /submissions/CIK{padded}.json → recent filings
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


@dataclass(frozen=True)
class Filing:
    accession: str
    form: str
    filed: str  # YYYY-MM-DD
    primary_doc: str
    ticker: str
    cik: int


def _headers() -> dict[str, str]:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "crag-fin research").strip()
    return {"User-Agent": ua, "Accept": "application/json"}


def _ticker_to_cik(ticker: str, session: requests.Session | None = None) -> int | None:
    s = session or requests.Session()
    resp = s.get(TICKERS_URL, headers=_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    needle = ticker.upper().strip()
    for entry in data.values():
        if str(entry.get("ticker", "")).upper() == needle:
            return int(entry["cik_str"])
    return None


def lookup_filings(
    ticker: str,
    n: int = 5,
    session: requests.Session | None = None,
) -> list[Filing]:
    """Return up to n recent filings for a ticker. Empty list on miss."""
    s = session or requests.Session()
    cik = _ticker_to_cik(ticker, session=s)
    if cik is None:
        return []
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    resp = s.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms = recent.get("form", [])
    filed = recent.get("filingDate", [])
    primaries = recent.get("primaryDocument", [])

    out: list[Filing] = []
    for i in range(min(n, len(accessions))):
        out.append(
            Filing(
                accession=accessions[i],
                form=forms[i] if i < len(forms) else "",
                filed=filed[i] if i < len(filed) else "",
                primary_doc=primaries[i] if i < len(primaries) else "",
                ticker=ticker.upper(),
                cik=cik,
            )
        )
    return out
