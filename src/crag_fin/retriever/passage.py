"""Passage dataclass shared by all retrievers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """A single retrieved passage.

    Returned by every retriever so downstream components (decomposer,
    reasoner, auditor) can stay retriever-agnostic.

    `credibility` and `weighted_score` are populated when the adapter is
    called with `weighted=True`; default to a credibility of 1.0 and a
    None weighted score so uniform retrieval results are unchanged.
    """

    text: str
    source: str
    score: float
    section: str = ""
    retrieval_method: str = ""
    credibility: float = 1.0
    weighted_score: float | None = None
