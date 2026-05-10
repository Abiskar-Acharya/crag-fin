# CRAG-FIN Architecture

## Five components

```
                       claim_text
                            │
                            ▼
                  ┌─────────────────────┐
                  │ 1. Decomposer       │  qwen2.5:7b (Ollama)
                  └─────────────────────┘
                            │
                  list[atomic_propositions]
                            │
                            ▼
                  ┌─────────────────────┐
                  │ 2. Retriever        │  BM25 + bge-large-en + RRF + cross-encoder
                  │   + credibility prior│  ChromaDB index + Tavily web search
                  └─────────────────────┘
                            │
              list[(proposition, evidence_with_credibility)]
                            │
                            ▼
                  ┌─────────────────────┐
                  │ 3. Agentic reasoner │  glm-4.7-flash:latest (Ollama)
                  └─────────────────────┘                multi-hop, up to 3 retrieval rounds
                            │
              list[(proposition, raw_verdict, justification, citations)]
                            │
                            ▼
                  ┌─────────────────────┐
                  │ 4. Calibrator       │  conformal prediction (mapie)
                  └─────────────────────┘                FN cost = 5× FP cost
                            │
              list[(proposition, calibrated_verdict, coverage)]
                            │
                            ▼
                       aggregator → final claim verdict + evidence chain

         ─────  separately, on top of the whole pipeline:  ─────

                  ┌─────────────────────┐
                  │ 5. Auditor (M4)     │  PRISM role-conditioning
                  └─────────────────────┘                hermes3:8b assessor
                            │
                  bias surface across personas
```

## Component contracts

### Decomposer
- **Input:** `claim: str`
- **Output:** `List[AtomicProposition]` where each has `{text, type, entity}`
- **Failure modes:** malformed JSON, claim already atomic, claim too long
- **Test invariants:** every output proposition is verifiable independently; no proposition is empty

### Retriever
- **Input:** `proposition: AtomicProposition`
- **Output:** `List[ScoredPassage]` where each has `{text, source_url, credibility_score, retrieval_score}`
- **Failure modes:** ChromaDB index empty, web search failed, no domain match in credibility prior
- **Test invariants:** top-1 passage always has source_url; credibility_score in [0,1]

### Reasoner
- **Input:** `proposition: AtomicProposition, evidence: List[ScoredPassage]`
- **Output:** `Verdict {verdict ∈ {TRUE, FALSE, UNCERTAIN}, justification: str, citations: List[int]}`
- **Failure modes:** structured output parse failure, citation index out of bounds, contradictory evidence
- **Test invariants:** every TRUE/FALSE verdict has at least one citation; UNCERTAIN may have zero

### Calibrator
- **Input:** `verdict_with_score: Verdict, calibration_set: List[CalibrationItem]`
- **Output:** `CalibratedVerdict {verdict ∈ {TRUE, FALSE, ABSTAIN}, prediction_set: Set[Verdict], coverage_at_alpha: float}`
- **Failure modes:** calibration set distribution shift, empty prediction set (bug), all-singleton prediction sets (under-calibrated)
- **Test invariants:** empirical coverage on test set within α ± 0.03 of nominal

### Auditor
- **Input:** `pipeline_callable, test_claims: List[str], role_library: List[Role]`
- **Output:** `BiasSurface {role × claim → verdict, drift_per_role_pair, kappa_across_roles}`
- **Failure modes:** assessor LLM disagrees with humans (low kappa), role prompts inconsistent across runs
- **Test invariants:** all 6 roles produce a verdict for every test claim (no silent failures)

## Data flow

```
ChromaDB ←──┐                            ┌──→ Local Ollama (qwen, hermes, glm)
            │                            │
            ▼                            ▼
       Retriever ──→ Reasoner ──→ Calibrator ──→ Output
            ▲                            ▲
            │                            │
       Tavily, SEC, FCA           Held-out calibration set
```

## Why this design

- **Separation of retrieval and reasoning** — so the retrieval can be updated (new news) without retraining the reasoner.
- **Explicit credibility prior** — so the same retrieved passage is weighted differently depending on source. Reuters > anonymous Telegram.
- **Conformal calibration** — so abstention is a statistical guarantee, not an ad-hoc threshold.
- **PRISM audit on the verifier (not the LLM advisor)** — direct extension of Moshfeghi's methodology to a deployed pipeline.

## What this is NOT

- A trading signal generator. CRAG-FIN classifies claims; it does not predict prices.
- A real-time streaming detector. The architecture is batch-friendly; streaming is a future engineering concern.
- A multimodal system. Text only.
- A general-purpose fact-checker. Tuned for UK-regulated financial English.
