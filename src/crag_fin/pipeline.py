"""End-to-end pipeline: claim -> decompose -> retrieve(weighted) -> reason.

Aggregation rule (Phase 1, placeholder for the conformal calibrator):
  - any proposition verdict == FALSE  -> claim FALSE
  - all proposition verdicts == TRUE  -> claim TRUE
  - otherwise                          -> claim ABSTAIN
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crag_fin.decomposer.decompose import Proposition, decompose
from crag_fin.reasoner.single_agent import Verdict, reason
from crag_fin.retriever.arxivmind_adapter import retrieve
from crag_fin.retriever.passage import Passage

ClaimVerdictLabel = str  # "TRUE" | "FALSE" | "ABSTAIN"


@dataclass(frozen=True)
class PropositionTrace:
    proposition: Proposition
    passages: list[Passage]
    verdict: Verdict


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    propositions: list[PropositionTrace] = field(default_factory=list)
    label: str = "ABSTAIN"
    reason_summary: str = ""


def _aggregate(verdicts: list[str]) -> str:
    if not verdicts:
        return "ABSTAIN"
    if "FALSE" in verdicts:
        return "FALSE"
    if all(v == "TRUE" for v in verdicts):
        return "TRUE"
    return "ABSTAIN"


def verify_claim(
    claim: str,
    *,
    k: int = 5,
    collection: str | None = "news",
    weighted: bool = True,
) -> ClaimVerdict:
    """Verify a financial claim end-to-end.

    1. Decompose into atomic propositions.
    2. Retrieve top-k passages per proposition (credibility-weighted by default).
    3. Reason over each (proposition, passages) pair.
    4. Aggregate per-proposition verdicts into a claim verdict.
    """
    propositions = decompose(claim)
    if not propositions:
        # Decomposer failed; treat the whole claim as one atomic proposition.
        propositions = [
            Proposition(proposition=claim, type="other", ticker_or_entity=None),
        ]

    traces: list[PropositionTrace] = []
    for prop in propositions:
        passages = retrieve(prop.proposition, k=k, weighted=weighted, collection=collection)
        verdict = reason(prop.proposition, passages)
        traces.append(PropositionTrace(proposition=prop, passages=passages, verdict=verdict))

    label = _aggregate([t.verdict.label for t in traces])
    if label == "UNCERTAIN":
        label = "ABSTAIN"

    reason_lines = [
        f"  [{i}] {t.proposition.proposition} -> {t.verdict.label}: {t.verdict.justification}"
        for i, t in enumerate(traces, 1)
    ]
    summary = f"{label}\n" + "\n".join(reason_lines)
    return ClaimVerdict(claim=claim, propositions=traces, label=label, reason_summary=summary)
