# ruff: noqa: E501  -- prompt content; long lines preserve example readability
"""Prompt templates for the single-agent reasoner.

Lifted verbatim from `~/Documents/Obsidian/Noosphere/006 Projects/CRAG-FIN/06_socratic_review.md`
section C1 (the "owed answer"). Output schema:

    {"verdict": "TRUE" | "FALSE" | "UNCERTAIN",
     "justification": "<2 sentences>",
     "citations": [1, 2, ...]}  // 1-indexed passage numbers

Day 3 wires this with GLM-4.7-Flash via Ollama. UNCERTAIN is a hard-threshold
abstention placeholder; Phase 7 (Day 19+) replaces it with conformal
prediction.
"""

from __future__ import annotations

REASONER_SYSTEM = """You are verifying a financial claim against retrieved evidence.

TASK:
1. Read each passage. State whether it SUPPORTS, CONTRADICTS, or is IRRELEVANT to the claim.
2. Output a verdict: TRUE, FALSE, or UNCERTAIN.
3. Output a 2-sentence justification citing specific passages by their number.
4. If evidence is contradictory, sparse, or off-topic, output UNCERTAIN.

OUTPUT FORMAT (STRICT JSON, no prose, no markdown):
{"verdict": "TRUE|FALSE|UNCERTAIN", "justification": "<2 sentences>", "citations": [<passage_number>, ...]}
"""

REASONER_USER_TEMPLATE = """CLAIM: {claim}

EVIDENCE:
{evidence_block}

Now produce the JSON verdict.
"""

PASSAGE_TEMPLATE = "[Passage {idx}] [Source: {source}, credibility_score: {cred:.2f}]\n{text}\n"
