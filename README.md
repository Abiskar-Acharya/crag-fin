# CRAG-FIN

A calibrated, source-credibility-weighted, retrieval-augmented verifier for financial claims. You give it a financial statement; it returns TRUE, FALSE, or ABSTAIN with cited evidence.

## What it does

The system reads a claim like *"GameStop bid $100B for eBay"* and decides whether the evidence supports it. It breaks the claim into atomic propositions, retrieves evidence weighted by how credible each source is, reasons over that evidence, and either commits to a verdict or abstains when it can't ground an answer.

The point is the abstention. In finance a wrong-but-confident answer is worse than "I don't know," so the design goal is to be right or silent rather than confidently wrong.

## Why it exists

Existing financial misinformation detectors (FMDLlama, FinFakeBERT) are static classifiers. They share three weaknesses: they don't know about anything after their training cutoff, they collapse when moved off their training distribution (the CN6000 RoBERTa baseline here scores F1 = 1.00 on ISOT but 0.41 on LIAR), and they give you a label with no evidence trail. A regulator can't act on "the model says fake."

CRAG-FIN swaps the static classifier for a retrieval-and-reasoning pipeline that cites its sources, weights those sources by credibility, and abstains under uncertainty.

## Research relevance

This is the working prototype behind my PhD proposal, *CRAG-FIN: Trustworthy Retrieval-Augmented Generation for High-Stakes Financial Question Answering*. It exists to back three claims in that proposal with running code:

- Evaluation beyond generic relevance. Finance needs metrics for faithfulness, numerical and temporal correctness, source credibility, and abstention quality (risk-coverage), not just "did the retrieved passage look relevant." The eval harness here is the first version of that benchmark.
- Verification. A claim-level layer that attributes each proposition to evidence and checks it, to cut down on unsupported or stale answers.
- Calibration. Uncertainty estimates that let the system defer on questions it can't ground. The current threshold-based abstention is a placeholder for a conformal calibrator (Phase 7).

It sits alongside my other portfolio work on memory and RAG reliability. Read together, the repos move from fundamentals to systems to evaluation.

## Current results (honest snapshot)

The only committed eval run is small and, frankly, unflattering: on 20 Fin-Fact claims the Phase-1 pipeline abstains on everything, so macro-F1 sits at 0.13 and accuracy at 0.25. Uniform and credibility-weighted retrieval score identically at this size.

| mode | n | macro-F1 | accuracy | abstention |
|------|---|----------|----------|------------|
| uniform | 20 | 0.133 | 0.250 | 100% |
| weighted | 20 | 0.133 | 0.250 | 100% |

That 100% abstention is the reasoner refusing to commit when the retrieved evidence doesn't clearly support or contradict a claim. Treat the table above as a baseline to beat, not a headline.

## How it works

A claim flows through five stages:

1. Decompose. Split the claim into atomic, independently checkable propositions.
2. Retrieve. Pull evidence from a ChromaDB index (hybrid BM25 + dense + reranking, reusing the ArXivMind retriever), then re-rank by a source-credibility prior over the domain.
3. Reason. Judge each proposition against its evidence and cite the passages used.
4. Calibrate. Wrap verdicts in an abstain decision. Currently a hard threshold; a conformal layer with an FCA-asymmetric loss is planned.
5. Audit. Re-run under role-conditioned prompts to measure verdict drift (PRISM-style). Scaffolded, not yet wired.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Quick start

```bash
# 1. Install
poetry install
cp .env.example .env        # add your keys

# 2. Pull local models (Ollama must be running)
ollama pull qwen2.5:7b      # decomposer
ollama pull glm-4.7-flash   # reasoner

# 3. Fetch data and seed the retrieval indexes
python scripts/setup_data.py

# 4. Verify a single claim
python -m crag_fin.pipeline "Apple is under FCA investigation for accounting fraud"

# 5. Run the tests
pytest

# 6. Run the Fin-Fact eval harness (small by default)
python -m experiments.scripts.run_finfact_eval --mode weighted --n 50 \
    --out experiments/runs/my_run/
python -m experiments.scripts.compute_metrics --run experiments/runs/my_run --mode weighted --date 2026-07-30
python -m experiments.scripts.make_plots --run experiments/runs/my_run
```

Everything above runs against local Ollama models. A pluggable backend (hosted DeepSeek/MiMo as drop-in alternatives) is in progress but not yet wired into the decomposer/reasoner — a near-term roadmap item, not a current option.

## Reproducibility, honestly

This does not yet clone-and-run from scratch, and I'd rather say so than pretend. Two real dependencies:

- Retrieval reuses the external [ArXivMind / glm-rag-pipeline](https://github.com/Abiskar-Acharya/glm-rag-pipeline) repo.
- Large data and vector indexes are gitignored. `python scripts/setup_data.py` fetches Fin-Fact and seeds the ChromaDB indexes; it's idempotent and skips work already done.

Runs are deterministic where it matters: all LLM calls use `temperature=0.0`, and each run writes a `manifest.json` recording mode, sample size, models, and elapsed time. Metrics are computed from the raw `predictions.jsonl` and saved before any plotting.

## Limitations

- The committed result is 20 claims and abstains on all of them. The prototype is honest about grounding but currently too conservative.
- Retrieval depends on an external repo and locally seeded indexes, so setup is more than one command.
- The credibility prior covers 30 sources across 4 tiers. It's a first pass.
- The calibrator and bias auditor are scaffolds. Abstention is a hard threshold, not calibrated.
- The hosted-LLM backend switch is not wired yet; the pipeline runs on local Ollama only today.
- This is not financial advice. It answers factual, grounded questions and abstains otherwise.

## Roadmap

- Re-run the eval after prompt and corpus-alignment fixes; commit real numbers and a confusion figure that isn't all-abstain.
- Wire a pluggable LLM backend so hosted providers (DeepSeek, MiMo) are drop-in alternatives to local Ollama.
- Wire the conformal calibrator and add a risk-coverage curve.
- Grow Fin-Fact into a stratified finance benchmark (static vs time-sensitive, lookup vs numerical).
- Ablate the verification and calibration layers separately; test robustness to retrieval noise and stale corpora.

## Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Fin-Fact | [Rangapur 2023](https://arxiv.org/abs/2309.08793) | Primary financial fact-checking benchmark |
| FinGuard | [carlos-gmartin](https://github.com/carlos-gmartin/Financial-Truth-Guard) | Secondary benchmark |
| LIAR | [Wang 2017](https://huggingface.co/datasets/liar) | Cross-domain political control |
| ISOT | UVic ISOT | Cross-domain news control |
| MBFC tier list | Media Bias / Fact Check | Source-credibility prior |
| UK FCA notices | FCA website | UK-regulated finance corpus (built locally) |

See [data/README.md](data/README.md) for provenance and licensing.

## Reuse

- Built on [ArXivMind](https://github.com/Abiskar-Acharya/glm-rag-pipeline) for the retrieval substrate.
- Conformal layer adapted from my [Conformal Safety ML](https://github.com/Abiskar-Acharya/conformal-safety-ml) work.
- PRISM audit methodology from [Azzopardi & Moshfeghi 2024](https://arxiv.org/abs/2410.18906).

## Author and license

Abiskar Acharya · abiskaracharya1@gmail.com · [github.com/Abiskar-Acharya](https://github.com/Abiskar-Acharya)

MIT License. See [LICENSE](LICENSE).

## Citing

```bibtex
@misc{acharya2026cragfin,
  title  = {{CRAG-FIN}: A Calibrated, Source-Credibility-Weighted, Agentic Verifier for Financial Misinformation},
  author = {Acharya, Abiskar},
  year   = {2026},
  note   = {Research prototype}
}
```
