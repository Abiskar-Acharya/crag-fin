"""FCA Register lookup — authorised firms by name.

FCA's public-facing register exposes a JSON API at
`https://register.fca.org.uk/services/V0.1/Firm/...`. Two-header auth
(X-Auth-Email + X-Auth-Key) is required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

BASE_URL = "https://register.fca.org.uk/services/V0.1/Firm"


@dataclass(frozen=True)
class FCAFirm:
    name: str
    frn: str  # Firm Reference Number
    status: str
    type_of_business: str
    business_country: str


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "X-Auth-Email": os.environ.get("FCA_REGISTER_AUTH_EMAIL", ""),
        "X-Auth-Key": os.environ.get("FCA_REGISTER_API_KEY", ""),
    }


def lookup_firm(name_or_frn: str, session: requests.Session | None = None) -> FCAFirm | None:
    """Return the top FCA-registered firm hit for the query, or None."""
    if not os.environ.get("FCA_REGISTER_API_KEY", "").strip():
        return None
    s = session or requests.Session()
    # FCA's V0.1 search endpoint takes either a firm name or FRN
    query = name_or_frn.strip()
    url = f"{BASE_URL}/{query}" if query.isdigit() else f"{BASE_URL}?q={query}"
    resp = s.get(url, headers=_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("Data", []) if isinstance(data, dict) else []
    if not items:
        return None
    item = items[0]
    return FCAFirm(
        name=str(item.get("Organisation Name") or item.get("Name", "")),
        frn=str(item.get("FRN", "") or item.get("Reference Number", "")),
        status=str(item.get("Status", "")),
        type_of_business=str(item.get("Type of business or Sub-category", "")),
        business_country=str(item.get("Country", "")),
    )
