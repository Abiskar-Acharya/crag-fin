# ruff: noqa: E501  -- prompt content; long lines preserve example readability
"""Prompt templates for the claim decomposer.

The decomposer breaks a compound financial claim into atomic propositions
that the retriever can score independently. We follow the LoCal (Ma 2025)
pattern: each proposition is verifiable against a single evidence span,
tagged with a coarse type for downstream retrieval routing, and pinned to
the primary entity (ticker or named legal entity).

Output schema (strict JSON list):
    [
      {
        "proposition": "<atomic verifiable statement>",
        "type": "earnings|regulatory|ESG|M&A|price_movement|other",
        "ticker_or_entity": "<TICKER or 'Entity Name' or null>"
      },
      ...
    ]

Maximum 5 propositions per claim. If the input is already atomic, return
a single-element list. Authored on Day 2; to be back-ported into the
vault note `02_socratic_questions.md` section C1 at session close.
"""

from __future__ import annotations

DECOMPOSER_SYSTEM = """You are a financial-claim decomposer. Your job is to split a compound \
financial claim into atomic, independently verifiable propositions.

Rules:
1. Each proposition must be checkable against a single piece of evidence.
2. Tag each proposition with one type from: earnings, regulatory, ESG, M&A, price_movement, other.
3. Identify the primary ticker (e.g. AAPL) or legal entity (e.g. "Barclays plc"). Use null when no specific entity is named.
4. Preserve quantitative details verbatim (numbers, dates, percentages).
5. Output STRICT JSON ONLY in this exact shape: {"propositions": [ {object}, {object}, ... ]}. The top-level value MUST be a JSON object containing a single key "propositions" whose value is an array. Each array element is an object with keys proposition, type, ticker_or_entity. No prose, no markdown, no code fences.
6. Maximum 5 propositions. If the claim is already atomic, the array still contains exactly one element.
"""

DECOMPOSER_USER_TEMPLATE = """Decompose the following financial claim into atomic propositions.

Examples:

Claim: "Apple beat Q3 earnings estimates and is under FCA investigation for accounting fraud."
Output: {{"propositions": [
  {{"proposition": "Apple beat Q3 earnings estimates.", "type": "earnings", "ticker_or_entity": "AAPL"}},
  {{"proposition": "Apple is under FCA investigation for accounting fraud.", "type": "regulatory", "ticker_or_entity": "AAPL"}}
]}}

Claim: "Tesla acquired SolarCity in 2016 for $2.6 billion, and the deal was approved by 85% of shareholders."
Output: {{"propositions": [
  {{"proposition": "Tesla acquired SolarCity in 2016 for $2.6 billion.", "type": "M&A", "ticker_or_entity": "TSLA"}},
  {{"proposition": "85% of shareholders approved the Tesla-SolarCity deal.", "type": "M&A", "ticker_or_entity": "TSLA"}}
]}}

Claim: "Barclays' share price rose 4% on Tuesday after the bank announced a £1bn buyback."
Output: {{"propositions": [
  {{"proposition": "Barclays' share price rose 4% on Tuesday.", "type": "price_movement", "ticker_or_entity": "Barclays plc"}},
  {{"proposition": "Barclays announced a £1bn share buyback.", "type": "M&A", "ticker_or_entity": "Barclays plc"}}
]}}

Now decompose this claim:

Claim: "{claim}"
Output:"""
