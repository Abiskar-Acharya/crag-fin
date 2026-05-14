"""Single-agent reasoner — verify one proposition against retrieved evidence.

Calls Ollama with `format="json"` + `temperature=0.0`. Returns a `Verdict`
with one of {TRUE, FALSE, UNCERTAIN}. UNCERTAIN is a hard-threshold
placeholder for the Phase-7 conformal calibrator.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal

import ollama

from crag_fin.reasoner.prompts import (
    PASSAGE_TEMPLATE,
    REASONER_SYSTEM,
    REASONER_USER_TEMPLATE,
)
from crag_fin.retriever.passage import Passage

VerdictLabel = Literal["TRUE", "FALSE", "UNCERTAIN"]
_VALID = {"TRUE", "FALSE", "UNCERTAIN"}
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True)
class Verdict:
    """Per-proposition verdict produced by the reasoner."""

    label: str  # TRUE / FALSE / UNCERTAIN
    justification: str
    citations: list[int]


def _build_evidence_block(passages: list[Passage]) -> str:
    if not passages:
        return "(no evidence retrieved)"
    parts: list[str] = []
    for i, p in enumerate(passages, 1):
        parts.append(
            PASSAGE_TEMPLATE.format(
                idx=i,
                source=(p.source or "unknown")[:80],
                cred=p.credibility,
                text=p.text.strip()[:600],
            )
        )
    return "\n".join(parts)


def _parse(raw: str) -> Verdict | None:
    text = raw.strip()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None
    label = str(parsed.get("verdict", "")).upper().strip()
    if label not in _VALID:
        return None
    justification = str(parsed.get("justification", "")).strip()
    cites_raw = parsed.get("citations", [])
    citations: list[int] = []
    if isinstance(cites_raw, list):
        for c in cites_raw:
            try:
                citations.append(int(c))
            except (TypeError, ValueError):
                continue
    return Verdict(label=label, justification=justification, citations=citations)


def reason(
    proposition: str,
    evidence: list[Passage],
    model: str | None = None,
) -> Verdict:
    """Produce a verdict for one proposition against retrieved passages.

    Empty proposition or empty evidence => UNCERTAIN with a placeholder
    justification (no LLM call). Parse failures => UNCERTAIN.
    """
    if not proposition.strip():
        return Verdict(label="UNCERTAIN", justification="Empty proposition.", citations=[])
    if not evidence:
        return Verdict(label="UNCERTAIN", justification="No evidence retrieved.", citations=[])

    model_name = model or os.environ.get("OLLAMA_REASONER_MODEL", "glm-4.7-flash:latest")
    user_msg = REASONER_USER_TEMPLATE.format(
        claim=proposition.strip(),
        evidence_block=_build_evidence_block(evidence),
    )

    for _ in range(2):
        response = ollama.chat(
            model=model_name,
            format="json",
            messages=[
                {"role": "system", "content": REASONER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.0},
        )
        verdict = _parse(response["message"]["content"])
        if verdict is not None:
            return verdict
    return Verdict(label="UNCERTAIN", justification="Reasoner parse failure.", citations=[])
