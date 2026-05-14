"""Unit tests for MBFC loader + scraper-parser."""

from __future__ import annotations

from pathlib import Path

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "mbfc_sample.csv"


def test_load_mbfc_tiers_round_trip() -> None:
    from crag_fin.retriever.mbfc_loader import load_mbfc_tiers

    out = load_mbfc_tiers(FIXTURE_CSV)
    assert out["reuters.com"] == 1.0
    assert out["bbc.co.uk"] == 0.8
    assert out["breitbart.com"] == 0.2
    assert out["infowars.com"] == 0.1
    # garbage label is skipped
    assert "example-unknown.com" not in out


def test_load_credibility_dict_merges_with_hand_coded() -> None:
    from crag_fin.retriever.mbfc_loader import load_credibility_dict

    merged = load_credibility_dict(FIXTURE_CSV)
    # MBFC-only entries present
    assert merged["breitbart.com"] == 0.2
    # Hand-coded entries override MBFC (reuters.com is in both)
    assert merged["reuters.com"] == 1.0
    # Hand-coded-only entries present
    assert "seekingalpha.com" in merged


def test_load_missing_csv_returns_hand_coded_only() -> None:
    from crag_fin.retriever.mbfc_loader import load_credibility_dict

    merged = load_credibility_dict(Path("/does/not/exist.csv"))
    assert "reuters.com" in merged  # hand-coded


def test_parse_outlet_page_extracts_fields() -> None:
    from crag_fin.tools.mbfc_scraper import parse_outlet_page

    html = """
    <html><body>
    <p>Bias Rating: LEFT-CENTER</p>
    <p>Factual Reporting: HIGH</p>
    <p>Source: https://www.bbc.co.uk</p>
    </body></html>
    """
    entry = parse_outlet_page(html, "https://mediabiasfactcheck.com/bbc/")
    assert entry is not None
    assert entry.domain == "bbc.co.uk"
    assert entry.factual_reporting == "HIGH"
    assert entry.bias_label == "LEFT-CENTER"


def test_parse_outlet_page_returns_none_on_missing_fields() -> None:
    from crag_fin.tools.mbfc_scraper import parse_outlet_page

    html = "<html><body>No useful labels here.</body></html>"
    assert parse_outlet_page(html, "https://mediabiasfactcheck.com/x/") is None
