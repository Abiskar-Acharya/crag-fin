"""Passage dataclass shared by all retrievers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """A single retrieved passage.

    Returned by every retriever so downstream components (decomposer,
    reasoner, auditor) can stay retriever-agnostic.
    """

    text: str
    source: str
    score: float
    section: str = ""
    retrieval_method: str = ""
