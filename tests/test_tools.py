"""Unit tests for web/API tools — mocked so they pass without live keys."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def test_tavily_returns_empty_without_key() -> None:
    from crag_fin.tools.web_search import tavily_search

    with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
        assert tavily_search("anything") == []


def test_tavily_parses_results_with_mocked_client() -> None:
    from crag_fin.tools import web_search

    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"title": "T1", "url": "https://www.reuters.com/x/y", "content": "snip"},
        ],
    }
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test"}, clear=False), \
         patch("tavily.TavilyClient", return_value=mock_client):
        hits = web_search.tavily_search("apple")
    assert len(hits) == 1
    assert hits[0].source_domain == "reuters.com"


def test_sec_edgar_ticker_lookup_mocked() -> None:
    from crag_fin.tools import sec_edgar

    tickers_response = MagicMock()
    tickers_response.json.return_value = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    tickers_response.raise_for_status = MagicMock()
    sub_response = MagicMock()
    sub_response.json.return_value = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000320193-26-000001"],
                "form": ["10-Q"],
                "filingDate": ["2026-05-01"],
                "primaryDocument": ["aapl10q.htm"],
            },
        },
    }
    sub_response.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.side_effect = [tickers_response, sub_response]

    filings = sec_edgar.lookup_filings("AAPL", n=1, session=session)
    assert len(filings) == 1
    assert filings[0].form == "10-Q"
    assert filings[0].cik == 320193


def test_companies_house_returns_none_without_key() -> None:
    from crag_fin.tools.companies_house import lookup_company

    with patch.dict(os.environ, {"COMPANIES_HOUSE_API_KEY": ""}, clear=False):
        assert lookup_company("Barclays") is None


def test_companies_house_parses_top_hit() -> None:
    from crag_fin.tools import companies_house

    resp = MagicMock()
    resp.json.return_value = {
        "items": [
            {
                "title": "BARCLAYS PLC",
                "company_number": "00048839",
                "company_status": "active",
                "company_type": "plc",
                "date_of_creation": "1896-07-20",
                "address": {
                    "address_line_1": "1 Churchill Place",
                    "locality": "London",
                    "postal_code": "E14 5HP",
                },
            },
        ],
    }
    resp.raise_for_status = MagicMock()
    session = MagicMock()
    session.get.return_value = resp

    with patch.dict(os.environ, {"COMPANIES_HOUSE_API_KEY": "test"}, clear=False):
        out = companies_house.lookup_company("Barclays", session=session)
    assert out is not None
    assert out.number == "00048839"
    assert "London" in out.address_line


def test_fca_register_returns_none_without_key() -> None:
    from crag_fin.tools.fca_register import lookup_firm

    with patch.dict(os.environ, {"FCA_REGISTER_API_KEY": ""}, clear=False):
        assert lookup_firm("Barclays Bank UK PLC") is None
