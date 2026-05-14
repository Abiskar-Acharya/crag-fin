"""Unit tests for the single-agent reasoner."""

from __future__ import annotations

import os
import shutil

import pytest

HAS_OLLAMA = shutil.which("ollama") is not None
HAS_MODEL = bool(os.environ.get("OLLAMA_REASONER_MODEL"))


def test_parse_extracts_verdict() -> None:
    from crag_fin.reasoner.single_agent import _parse

    raw = '{"verdict": "TRUE", "justification": "Passage 1 confirms.", "citations": [1]}'
    v = _parse(raw)
    assert v is not None
    assert v.label == "TRUE"
    assert v.citations == [1]


def test_parse_rejects_invalid_label() -> None:
    from crag_fin.reasoner.single_agent import _parse

    raw = '{"verdict": "MAYBE", "justification": "?", "citations": []}'
    assert _parse(raw) is None


def test_parse_recovers_from_noise() -> None:
    from crag_fin.reasoner.single_agent import _parse

    raw = (
        'Here is your answer: '
        '{"verdict": "FALSE", "justification": "Contradicted by P2.", "citations": [2]}'
        ' (done)'
    )
    v = _parse(raw)
    assert v is not None
    assert v.label == "FALSE"


def test_reason_empty_evidence_abstains() -> None:
    from crag_fin.reasoner.single_agent import reason

    v = reason("Some claim.", [])
    assert v.label == "UNCERTAIN"


@pytest.mark.skipif(
    not (HAS_OLLAMA and HAS_MODEL),
    reason="Ollama binary or OLLAMA_REASONER_MODEL env var missing",
)
def test_reason_live_with_supportive_passage() -> None:
    from crag_fin.reasoner.single_agent import reason
    from crag_fin.retriever.passage import Passage

    passages = [
        Passage(
            text=(
                "Apple Inc reported quarterly earnings of $1.65 per share, "
                "beating analyst estimates of $1.40."
            ),
            source="reuters.com",
            score=1.0,
            credibility=1.0,
        ),
    ]
    v = reason("Apple beat Q3 earnings estimates.", passages)
    assert v.label in {"TRUE", "FALSE", "UNCERTAIN"}
