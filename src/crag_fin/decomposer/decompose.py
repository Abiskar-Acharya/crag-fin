"""Compound-claim decomposer.

Calls a local Ollama model with `format="json"` to split a financial claim
into atomic propositions. Returns a list of `Proposition` dataclasses.
One JSON-parse retry on failure; if both attempts fail, returns an empty
list (caller decides how to handle).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal

import ollama

from crag_fin.decomposer.prompts import DECOMPOSER_SYSTEM, DECOMPOSER_USER_TEMPLATE

PropositionType = Literal["earnings", "regulatory", "ESG", "M&A", "price_movement", "other"]

MAX_PROPOSITIONS = 5
_JSON_LIST_RE = re.compile(r"\[[\s\S]*\]")


@dataclass(frozen=True)
class Proposition:
    """One atomic, independently verifiable proposition extracted from a claim."""

    proposition: str
    type: str
    ticker_or_entity: str | None


def _parse(raw: str) -> list[Proposition]:
    """Parse model output into a list of Proposition.

    Tries strict json.loads first; if that fails, tries the first JSON-list
    substring in the output. Returns [] on any structural problem rather
    than raising — the caller (notebook, pipeline) is in a better place
    to decide retry policy.
    """
    text = raw.strip()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_LIST_RE.search(text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    # Ollama format="json" often returns a single object even when the
    # prompt asks for an array. Wrap single dicts in a list. Also unwrap
    # the common {"propositions": [...]} envelope.
    if isinstance(parsed, dict):
        if "propositions" in parsed and isinstance(parsed["propositions"], list):
            parsed = parsed["propositions"]
        elif "proposition" in parsed:
            parsed = [parsed]
        else:
            return []
    if not isinstance(parsed, list):
        return []

    out: list[Proposition] = []
    for item in parsed[:MAX_PROPOSITIONS]:
        if not isinstance(item, dict):
            continue
        prop = item.get("proposition")
        typ = item.get("type", "other")
        entity = item.get("ticker_or_entity")
        if not isinstance(prop, str) or not prop.strip():
            continue
        out.append(
            Proposition(
                proposition=prop.strip(),
                type=str(typ) if typ else "other",
                ticker_or_entity=str(entity) if entity else None,
            )
        )
    return out


def decompose(claim: str, model: str | None = None) -> list[Proposition]:
    """Decompose a financial claim into atomic propositions.

    Args:
        claim: The compound (or atomic) financial claim.
        model: Ollama model name. Defaults to OLLAMA_DECOMPOSER_MODEL env var.

    Returns:
        List of Proposition (length 0..5). Empty list signals a parse failure;
        the caller decides whether to retry, fall back, or abstain.
    """
    if not claim or not claim.strip():
        return []

    model_name = model or os.environ.get("OLLAMA_DECOMPOSER_MODEL", "qwen2.5:7b")
    user_msg = DECOMPOSER_USER_TEMPLATE.format(claim=claim.strip())

    for _attempt in range(2):
        response = ollama.chat(
            model=model_name,
            format="json",
            messages=[
                {"role": "system", "content": DECOMPOSER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 0.0},
        )
        content = response["message"]["content"]
        parsed = _parse(content)
        if parsed:
            return parsed
    return []
