"""Smoke tests for src/crag_fin/retriever/."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("ARXIVMIND_CHROMA_DIR", "")).exists(),
    reason="ArXivMind chroma dir not available on this machine",
)


def test_passage_dataclass_round_trip() -> None:
    from crag_fin.retriever.passage import Passage

    p = Passage(text="hello", source="test", score=0.5)
    assert p.text == "hello"
    assert p.section == ""
    assert p.credibility == 1.0
    assert p.weighted_score is None


def test_passage_carries_credibility() -> None:
    from crag_fin.retriever.passage import Passage

    p = Passage(text="t", source="reuters.com", score=0.8, credibility=1.0, weighted_score=0.8)
    assert p.weighted_score == 0.8


def test_arxivmind_retrieve_returns_at_most_k() -> None:
    """retrieve() returns a list of <= k Passages."""
    from crag_fin.retriever.arxivmind_adapter import retrieve
    from crag_fin.retriever.passage import Passage

    results = retrieve("attention mechanism transformer", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert isinstance(r, Passage)
        assert isinstance(r.text, str)
        assert isinstance(r.score, float)
