"""UK Companies House lookup — company search by name or number.

API auth is HTTP Basic with the API key as the username and an empty
password. Set COMPANIES_HOUSE_API_KEY in .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

SEARCH_URL = "https://api.company-information.service.gov.uk/search/companies"


@dataclass(frozen=True)
class Company:
    name: str
    number: str  # Companies House number (e.g. "06012345")
    status: str  # "active", "dissolved", ...
    company_type: str
    incorporated_on: str  # ISO date or ""
    address_line: str


def lookup_company(query: str, session: requests.Session | None = None) -> Company | None:
    """Search Companies House and return the top hit, or None on miss / no key."""
    api_key = os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip()
    if not api_key:
        return None
    s = session or requests.Session()
    resp = s.get(
        SEARCH_URL,
        params={"q": query, "items_per_page": 1},
        auth=(api_key, ""),
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return None
    item = items[0]
    address = item.get("address", {}) or {}
    address_parts = [
        address.get("address_line_1"),
        address.get("locality"),
        address.get("postal_code"),
    ]
    return Company(
        name=str(item.get("title", "")),
        number=str(item.get("company_number", "")),
        status=str(item.get("company_status", "")),
        company_type=str(item.get("company_type", "")),
        incorporated_on=str(item.get("date_of_creation", "")),
        address_line=", ".join(v for v in address_parts if v),
    )
