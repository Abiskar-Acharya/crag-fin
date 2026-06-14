"""Compute Fin-Fact metrics for a finished run and append a row to the results CSV.

Reads a run's predictions.jsonl, writes metrics.json beside it (raw numbers
saved before any reporting, per the reproducibility rule), and appends one row
to experiments/results/retriever_v1.csv.

Usage:
    python -m experiments.scripts.compute_metrics \
        --run experiments/runs/run_002_aligned/uniform_n20_sharpened \
        --mode uniform --date 2026-06-14
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from crag_fin.eval.metrics import compute_finfact_metrics

CSV_HEADER = ["run", "mode", "n", "macro_f1", "accuracy", "abstention_rate", "date"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run dir containing predictions.jsonl")
    parser.add_argument("--mode", choices=["uniform", "weighted"], required=True)
    parser.add_argument("--date", required=True, help="ISO date for the CSV row")
    parser.add_argument("--csv", default="experiments/results/retriever_v1.csv")
    args = parser.parse_args()

    run_dir = Path(args.run)
    preds = run_dir / "predictions.jsonl"
    metrics = compute_finfact_metrics(preds)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    csv_path = Path(args.csv)
    write_header = not csv_path.exists()
    # run label is the path relative to experiments/runs for readability
    run_label = str(run_dir).replace("experiments/runs/", "")
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerow(
            [
                run_label,
                args.mode,
                metrics["n"],
                f"{metrics['macro_f1']:.3f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['abstention_rate']:.3f}",
                args.date,
            ]
        )

    print(f"[metrics] {run_label}: macro_f1={metrics['macro_f1']:.3f} "
          f"acc={metrics['accuracy']:.3f} abstain={metrics['abstention_rate']:.3f} "
          f"n={metrics['n']}")
    print(f"[metrics] pred_dist={metrics['pred_dist']} gold_dist={metrics['gold_dist']}")


if __name__ == "__main__":
    main()
