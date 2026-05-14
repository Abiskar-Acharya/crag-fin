"""Unit tests for the hand-coded credibility prior."""

from __future__ import annotations

from crag_fin.retriever.credibility_prior import (
    DEFAULT_CREDIBILITY,
    credibility,
    extract_domain,
)


def test_extract_domain_strips_scheme_and_www() -> None:
    assert extract_domain("https://www.reuters.com/article/x") == "reuters.com"
    assert extract_domain("http://bloomberg.com/news/y") == "bloomberg.com"


def test_extract_domain_handles_bare_domain() -> None:
    assert extract_domain("reuters.com") == "reuters.com"
    assert extract_domain("WWW.BBC.CO.UK") == "bbc.co.uk"


def test_extract_domain_handles_subpath() -> None:
    assert extract_domain("reuters.com/article/foo/bar") == "reuters.com"


def test_extract_domain_empty_returns_empty() -> None:
    assert extract_domain("") == ""


def test_credibility_tier1() -> None:
    assert credibility("https://www.reuters.com/x/y") == 1.0
    assert credibility("ft.com") == 1.0


def test_credibility_tier3() -> None:
    assert credibility("https://seekingalpha.com/article/123") == 0.4


def test_credibility_unknown_domain_is_default() -> None:
    assert credibility("https://random-blog-12345.example") == DEFAULT_CREDIBILITY


def test_credibility_empty_source_is_default() -> None:
    assert credibility("") == DEFAULT_CREDIBILITY


def test_credibility_capitalisation_invariant() -> None:
    assert credibility("HTTPS://Reuters.COM/x") == 1.0
