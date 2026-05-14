"""Unit tests for the tldextract-based domain extractor."""

from __future__ import annotations

from crag_fin.retriever.domain_extract import extract_domain, is_shortener


def test_plain_url() -> None:
    assert extract_domain("https://www.reuters.com/article/x") == "reuters.com"


def test_subdomain_collapses() -> None:
    assert extract_domain("https://uk.reuters.com/markets") == "reuters.com"
    assert extract_domain("https://regional.bbc.co.uk/scotland") == "bbc.co.uk"


def test_amp_page_subdomain() -> None:
    assert extract_domain("https://amp.bbc.co.uk/news/foo") == "bbc.co.uk"


def test_amp_page_path() -> None:
    # AMP-in-path is just a path; the host is unchanged.
    assert extract_domain("https://www.bbc.co.uk/amp/news/foo") == "bbc.co.uk"


def test_bare_domain_no_scheme() -> None:
    assert extract_domain("reuters.com") == "reuters.com"
    assert extract_domain("reuters.com/article/foo") == "reuters.com"


def test_capitalisation_invariant() -> None:
    assert extract_domain("HTTPS://WWW.REUTERS.COM/X") == "reuters.com"


def test_shortener_returns_none() -> None:
    assert extract_domain("https://bit.ly/abc123") is None
    assert extract_domain("https://t.co/abc") is None
    assert extract_domain("https://lnkd.in/abc") is None


def test_is_shortener_helper() -> None:
    assert is_shortener("https://bit.ly/x")
    assert is_shortener("bit.ly/x")
    assert not is_shortener("https://reuters.com/x")


def test_empty_and_bad_input() -> None:
    assert extract_domain("") is None
    assert extract_domain("   ") is None
    assert extract_domain("not-a-domain-no-tld") is None


def test_url_in_text() -> None:
    text = "See the Reuters article at https://www.reuters.com/article/123 for details."
    assert extract_domain(text) == "reuters.com"


def test_country_code_double_tld() -> None:
    assert extract_domain("https://www.theguardian.co.uk/x") == "theguardian.co.uk"


def test_trailing_punctuation() -> None:
    # tldextract handles trailing punctuation in the path naturally because
    # we split off the path before extraction.
    assert extract_domain("https://www.bloomberg.com/news/article-y.") == "bloomberg.com"
