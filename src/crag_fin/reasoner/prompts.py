# ruff: noqa: E501  -- prompt content; long lines preserve example readability
"""Prompt templates for the single-agent reasoner.

Originally lifted verbatim from `06_socratic_review.md` section C1. Output schema:

    {"verdict": "TRUE" | "FALSE" | "UNCERTAIN",
     "justification": "<2 sentences>",
     "citations": [1, 2, ...]}  // 1-indexed passage numbers

Day 3 wires this with GLM-4.7-Flash via Ollama. UNCERTAIN is a hard-threshold
abstention placeholder; Phase 7 (Day 19+) replaces it with conformal
prediction.

Day-12 sharpen (Session 012): the Day-11 aligned-corpus diagnostic showed the
reasoner abstaining on evidence that was topically aligned and substantively
on-point but did not restate the claim verbatim (the "relevant-but-not-pointed"
failure mode — see Session-011 open question 1). The system prompt below now
tells the model to reason about what a passage *implies* rather than whether it
lexically matches, and reserves UNCERTAIN for genuinely off-topic or
irreconcilably-conflicting evidence. The JSON output schema is unchanged.
"""

from __future__ import annotations

REASONER_SYSTEM = """You are verifying a financial claim against retrieved evidence.

TASK:
1. Read each passage. Decide whether it SUPPORTS, CONTRADICTS, or is IRRELEVANT to the claim. A passage supports or contradicts the claim if it bears on whether the claim is true — it need NOT restate the claim word-for-word or name every entity. Reason about what the passage implies, not whether it lexically matches.
2. Output a verdict:
   - TRUE  — at least one relevant passage supports the claim and none contradict it.
   - FALSE — at least one relevant passage contradicts the claim.
   - UNCERTAIN — only when the evidence genuinely does not let you decide: every passage is off-topic, OR relevant passages point in conflicting directions with no way to adjudicate between them.
3. Do NOT output UNCERTAIN merely because a passage is paraphrased, partial, or approximate. If a passage substantively supports or contradicts the claim, commit to TRUE or FALSE.
4. Output a 2-sentence justification citing specific passages by their number.

OUTPUT FORMAT (STRICT JSON, no prose, no markdown):
{"verdict": "TRUE|FALSE|UNCERTAIN", "justification": "<2 sentences>", "citations": [<passage_number>, ...]}
"""

REASONER_USER_TEMPLATE = """CLAIM: {claim}

EVIDENCE:
{evidence_block}

Now produce the JSON verdict.
"""

PASSAGE_TEMPLATE = "[Passage {idx}] [Source: {source}, credibility_score: {cred:.2f}]\n{text}\n"
