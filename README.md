# CRAG-FIN

A **C**alibrated, source-credibility-weighted, **R**etrieval-**A**ugmented, a**G**entic verifier for **FIN**ancial misinformation.

> **Status:** Prototype, May 2026. Not for production use.

## What this is

The system reads a financial claim — for example, *"Apple is under FCA investigation for accounting fraud"* — and outputs **TRUE / FALSE / ABSTAIN** with cited evidence and a calibrated confidence score.

It does this through five components: a decomposer, a source-credibility-weighted retriever, an agentic reasoner, a conformal calibrator, and a PRISM-style bias auditor.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Why this exists

Existing financial misinformation detectors (FMDLlama, FinFakeBERT) are static classifiers. They have three compounding problems: temporal cutoff (they don't know about events after training), cross-domain failure (CN6000 RoBERTa: F1 = 1.000 on ISOT → 0.407 on LIAR), and no evidence trail (a regulator can't act on "the model says fake").

CRAG-FIN closes a five-way integration gap: financial-domain agentic verification, source-credibility-weighted retrieval, conformal abstention with FCA-asymmetric loss, adversarial robustness against LLM-generated financial misinformation, and PRISM-style bias audit of the verifier itself.

## Quick start

```bash
# 1. Install
poetry install
cp .env.example .env  # add your Tavily / OpenAI / etc keys

# 2. Pull local models (Ollama must be running)
ollama pull glm-4.7-flash
ollama pull qwen2.5:7b
ollama pull hermes3:8b

# 3. Download datasets
python scripts/setup_data.py

# 4. Run end-to-end on a single claim
python -m crag_fin.pipeline "Apple is under FCA investigation for accounting fraud"

# 5. Run the test suite
pytest

# 6. Run the full evaluation harness on Fin-Fact
python -m crag_fin.eval.runner --dataset finfact --split test
```

## Architecture (one paragraph)

A financial claim flows through: (1) a **decomposer** (qwen2.5:7b via Ollama) that splits it into atomic verifiable propositions following the LoCal pattern; (2) a **retriever** (BM25 + dense bge-large-en + RRF + cross-encoder reranking, with a credibility prior over the source domain) that pulls evidence from a ChromaDB index plus live web tools; (3) an **agentic reasoner** (glm-4.7-flash via Ollama) that issues per-proposition verdicts with citations; (4) a **conformal calibrator** that wraps verdicts in prediction sets at user-chosen α with FCA-asymmetric loss; (5) a **PRISM-style auditor** that re-runs the pipeline under role-conditioned prompts and measures verdict drift across roles.

## Reuse

- Built on [ArXivMind](https://github.com/Abiskar-Acharya/glm-rag-pipeline) retrieval substrate
- Uses CN6000 RoBERTa as a baseline classifier
- Conformal layer adapted from [Conformal Safety ML](https://github.com/Abiskar-Acharya/conformal-safety-ml) project
- PRISM methodology from [Azzopardi & Moshfeghi 2024](https://arxiv.org/abs/2410.18906) (Strathclyde NeuraSearch Lab)

## Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Fin-Fact | [Rangapur 2023](https://arxiv.org/abs/2309.08793) | Primary FMD benchmark |
| FinGuard | [carlos-gmartin](https://github.com/carlos-gmartin/Financial-Truth-Guard) | Secondary FMD benchmark |
| LIAR | [Wang 2017](https://huggingface.co/datasets/liar) | Cross-domain political control |
| ISOT | UVic ISOT | Cross-domain news control |
| MBFC tier list | Media Bias / Fact Check | Source credibility prior |
| UK FCA enforcement notices | FCA website | UK-regulated finance corpus (built locally) |

## Author and license

Abiskar Acharya  ·  abiskaracharya1@gmail.com  ·  [github.com/Abiskar-Acharya](https://github.com/Abiskar-Acharya)

MIT License. See [LICENSE](LICENSE).

## Citing

If this work helps yours, please cite the (forthcoming) preprint:

```bibtex
@misc{acharya2026cragfin,
  title  = {{CRAG-FIN}: A Calibrated, Source-Credibility-Weighted, Agentic Verifier for Financial Misinformation},
  author = {Acharya, Abiskar},
  year   = {2026},
  eprint = {forthcoming},
  archivePrefix = {arXiv}
}
```
