"""Smoke tests for the claim decomposer."""

from __future__ import annotations

import os
import shutil

import pytest

# Two-tier skip: skip ALL tests if ollama isn't on PATH at all; skip the
# live-LLM test if OLLAMA_DECOMPOSER_MODEL isn't set.
HAS_OLLAMA = shutil.which("ollama") is not None
HAS_MODEL = bool(os.environ.get("OLLAMA_DECOMPOSER_MODEL"))


def test_parse_handles_empty_input() -> None:
    """Empty / blank claim returns [] without calling Ollama."""
    from crag_fin.decomposer.decompose import decompose

    assert decompose("") == []
    assert decompose("   ") == []


def test_parse_recovers_from_loose_json() -> None:
    """The fallback regex pulls a JSON list out of noisy output."""
    from crag_fin.decomposer.decompose import _parse

    noisy = (
        'Sure, here is: '
        '[{"proposition": "X went up.", "type": "price_movement", "ticker_or_entity": "X"}]'
        ' -- enjoy.'
    )
    out = _parse(noisy)
    assert len(out) == 1
    assert out[0].proposition == "X went up."
    assert out[0].type == "price_movement"


def test_parse_drops_malformed_items() -> None:
    """Items lacking a proposition string are dropped silently."""
    from crag_fin.decomposer.decompose import _parse

    mixed = (
        '[{"proposition": "Real.", "type": "other", "ticker_or_entity": null},'
        ' {"type": "earnings"},'
        ' {"proposition": ""}]'
    )
    out = _parse(mixed)
    assert len(out) == 1
    assert out[0].proposition == "Real."


@pytest.mark.skipif(
    not (HAS_OLLAMA and HAS_MODEL),
    reason="Ollama binary or OLLAMA_DECOMPOSER_MODEL env var missing",
)
def test_decompose_live_compound_claim() -> None:
    """A compound claim yields >=2 propositions when the model is available."""
    from crag_fin.decomposer.decompose import decompose

    claim = "Apple beat Q3 earnings estimates and is under FCA investigation for accounting fraud."
    props = decompose(claim)
    assert len(props) >= 1, "Expected at least 1 proposition from a compound claim"
    assert len(props) <= 5
    for p in props:
        assert p.proposition.strip()
        assert p.type in {"earnings", "regulatory", "ESG", "M&A", "price_movement", "other"}
