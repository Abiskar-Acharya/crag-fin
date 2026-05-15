"""Run the CRAG-FIN pipeline over Fin-Fact and save predictions for scoring.

Usage:
    python -m experiments.scripts.run_finfact_eval --mode weighted --n 50 \
        --out experiments/runs/run_001_uniform_vs_weighted/weighted_n50/

Outputs:
    {out}/predictions.jsonl   one line per claim with {claim, gold, pred, propositions, ...}
    {out}/manifest.json       run config (mode, n, k, models, timestamp)

Day-10 plan note: a full 1304-claim run takes hours on M4 with
glm-4.7-flash. The default --n is small so the harness is usable; bump
--n to 1304 for the publication run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["uniform", "weighted"], required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--finfact", default="data/finfact/raw/finfact.json")
    parser.add_argument(
        "--collection",
        choices=["news", "finfact"],
        default="finfact",
        help="news = crag_fin_news_v0 (Day 2 seed); finfact = finfact_evidence_v0 (Day 11 aligned)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Force HF cache-only so the underlying SentenceTransformer / cross-encoder
    # don't silently hang on an unauthenticated rate-limit retry. The models
    # are already cached locally from Day 1-2 runs.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    load_dotenv()
    # Import after .env so OLLAMA_* models are visible to the pipeline.
    from crag_fin.pipeline import verify_claim

    finfact = json.loads(Path(args.finfact).read_text())
    claims = finfact[: args.n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    started = time.time()
    weighted = args.mode == "weighted"
    with open(predictions_path, "w") as f:
        for i, row in enumerate(claims, 1):
            claim = row["claim"]
            gold = row.get("label", "NEI")
            cv = verify_claim(claim, k=args.k, weighted=weighted, collection=args.collection)
            record = {
                "i": i,
                "claim": claim,
                "gold": gold,
                "pred": cv.label,
                "propositions": [
                    {
                        "proposition": t.proposition.proposition,
                        "type": t.proposition.type,
                        "ticker_or_entity": t.proposition.ticker_or_entity,
                        "verdict": t.verdict.label,
                        "justification": t.verdict.justification,
                        "citations": t.verdict.citations,
                        "passage_sources": [p.source for p in t.passages],
                    }
                    for t in cv.propositions
                ],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()  # let `tail -f` see progress without waiting for 8KB buffer
            elapsed = time.time() - started
            if i % 5 == 0 or i == len(claims):
                print(
                    f"  [{i}/{len(claims)}] {elapsed:6.1f}s | last={cv.label} | {claim[:60]}",
                    flush=True,
                )

    manifest = {
        "mode": args.mode,
        "n": len(claims),
        "k": args.k,
        "weighted": weighted,
        "collection": args.collection,
        "finfact_path": args.finfact,
        "decomposer_model": os.environ.get("OLLAMA_DECOMPOSER_MODEL"),
        "reasoner_model": os.environ.get("OLLAMA_REASONER_MODEL"),
        "elapsed_sec": time.time() - started,
        "predictions_path": str(predictions_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {len(claims)} predictions to {predictions_path}")
    print(f"[done] manifest at {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
