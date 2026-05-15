"""Two-pass Fin-Fact eval that avoids per-claim Ollama model swapping.

Day 11 discovery: a 40 GB glm-4.7-flash + a 4.7 GB qwen2.5:7b combined exceed
the M4's GPU VRAM budget, so Ollama swaps one out to load the other on every
verify_claim() call. That swap dominated wall time (~3 min/claim vs ~25 s
on Day 1 when only one model was hot).

This script splits the work into two passes:
  PASS 1 — decompose all N claims with qwen2.5:7b (~5 s/claim).
  PASS 2 — retrieve + reason for every (claim, proposition) pair using
           glm-4.7-flash (~15 s/proposition).

Total wall time: roughly the sum of the two passes, no swap overhead.

Usage:
    python -m experiments.scripts.run_finfact_eval_batched \
        --mode {uniform,weighted} --n 20 --k 4 \
        --collection finfact --out experiments/runs/run_002_aligned/uniform_n20/
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
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--finfact", default="data/finfact/raw/finfact.json")
    parser.add_argument("--collection", choices=["news", "finfact"], default="finfact")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    load_dotenv()

    from crag_fin.decomposer.decompose import Proposition, decompose
    from crag_fin.reasoner.single_agent import reason
    from crag_fin.retriever.arxivmind_adapter import retrieve

    finfact = json.loads(Path(args.finfact).read_text())
    claims = finfact[: args.n]
    weighted = args.mode == "weighted"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    # ----- PASS 1: decompose -----
    print(f"[pass 1/2] decomposing {len(claims)} claims with qwen2.5:7b", flush=True)
    t0 = time.time()
    per_claim_props: list[list[Proposition]] = []
    for i, row in enumerate(claims, 1):
        props = decompose(row["claim"])
        if not props:
            # Fall back to the atomic claim itself so the reasoner pass still runs.
            props = [Proposition(proposition=row["claim"], type="other", ticker_or_entity=None)]
        per_claim_props.append(props)
        if i % 5 == 0 or i == len(claims):
            print(f"  decomp [{i}/{len(claims)}] {time.time() - t0:.1f}s", flush=True)
    decomp_secs = time.time() - t0

    # ----- PASS 2: retrieve + reason -----
    print("[pass 2/2] reasoning per proposition with glm-4.7-flash", flush=True)
    t1 = time.time()
    with open(predictions_path, "w") as f:
        for i, (row, props) in enumerate(zip(claims, per_claim_props, strict=True), 1):
            claim_traces = []
            for prop in props:
                passages = retrieve(
                    prop.proposition, k=args.k, weighted=weighted, collection=args.collection
                )
                verdict = reason(prop.proposition, passages)
                claim_traces.append(
                    {
                        "proposition": prop.proposition,
                        "type": prop.type,
                        "ticker_or_entity": prop.ticker_or_entity,
                        "verdict": verdict.label,
                        "justification": verdict.justification,
                        "citations": verdict.citations,
                        "passage_sources": [p.source for p in passages],
                    }
                )
            # Aggregate
            verdicts = [t["verdict"] for t in claim_traces]
            if "FALSE" in verdicts:
                final = "FALSE"
            elif all(v == "TRUE" for v in verdicts):
                final = "TRUE"
            else:
                final = "ABSTAIN"
            rec = {
                "i": i,
                "claim": row["claim"],
                "gold": row.get("label", "NEI"),
                "pred": final,
                "propositions": claim_traces,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(claims):
                print(
                    f"  reason [{i}/{len(claims)}] {time.time() - t1:.1f}s pred={final}",
                    flush=True,
                )
    reason_secs = time.time() - t1

    manifest = {
        "mode": args.mode,
        "n": len(claims),
        "k": args.k,
        "weighted": weighted,
        "collection": args.collection,
        "decomp_secs": decomp_secs,
        "reason_secs": reason_secs,
        "total_secs": decomp_secs + reason_secs,
        "decomposer_model": os.environ.get("OLLAMA_DECOMPOSER_MODEL"),
        "reasoner_model": os.environ.get("OLLAMA_REASONER_MODEL"),
        "predictions_path": str(predictions_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = decomp_secs + reason_secs
    print(
        f"[done] decomp={decomp_secs:.1f}s reason={reason_secs:.1f}s total={total:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
