#!/usr/bin/env python3
"""Turn committed eval outputs into publishable figures.

Reads what an eval run leaves behind and writes PNGs to ``paper/figures/``:

  1. ``confusion.png``        - gold vs. predicted labels for one run.
  2. ``uniform_vs_weighted.png`` - macro-F1 / accuracy / abstention by
     retrieval mode, from ``experiments/results/retriever_v1.csv``.
  3. ``risk_coverage.png``    - ONLY if predictions carry a per-item
     ``confidence`` field. Skipped with a note otherwise (the Phase-1
     pipeline does not emit calibrated confidence yet).

Everything here is honest about missing pieces: it plots what the data
supports and prints a clear message for what it cannot yet plot. Re-run it
after a fresh eval to regenerate the figures the README embeds.

Usage:
    python -m experiments.scripts.make_plots \
        --run experiments/runs/run_001_uniform_vs_weighted/weighted_n20 \
        --csv experiments/results/retriever_v1.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crag_fin.eval.metrics import GOLD_TO_PRED, LABELS, compute_finfact_metrics  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"


def _read_rows(predictions_jsonl: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(predictions_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold_raw = str(obj.get("gold", "")).strip()
            pred = str(obj.get("pred", "")).strip().upper()
            gold = GOLD_TO_PRED.get(gold_raw, GOLD_TO_PRED.get(gold_raw.lower(), gold_raw))
            if gold in LABELS and pred in LABELS:
                rows.append((gold, pred))
    return rows


def plot_confusion(run_dir: Path) -> None:
    preds = run_dir / "predictions.jsonl"
    if not preds.exists():
        print(f"  confusion: no predictions.jsonl in {run_dir} -> skip")
        return
    rows = _read_rows(preds)
    if not rows:
        print(f"  confusion: {preds} has no scorable rows -> skip")
        return

    idx = {label: i for i, label in enumerate(LABELS)}
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for gold, pred in rows:
        matrix[idx[gold]][idx[pred]] += 1

    metrics = compute_finfact_metrics(preds)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), labels=LABELS)
    ax.set_yticks(range(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > max(1, len(rows) // 6) else "black")
    ax.set_title(
        f"CRAG-FIN on Fin-Fact (n={metrics['n']})\n"
        f"macro-F1={metrics['macro_f1']:.3f}  acc={metrics['accuracy']:.3f}  "
        f"abstain={metrics['abstention_rate']:.0%}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIG_DIR / "confusion.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  confusion -> {out.relative_to(REPO_ROOT)}")


def plot_uniform_vs_weighted(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"  uniform_vs_weighted: {csv_path} missing -> skip")
        return
    import csv as _csv

    by_mode: dict[str, dict[str, float]] = {}
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            by_mode[row["mode"]] = {
                "macro_f1": float(row["macro_f1"]),
                "accuracy": float(row["accuracy"]),
                "abstention_rate": float(row["abstention_rate"]),
            }
    if not by_mode:
        print("  uniform_vs_weighted: no rows -> skip")
        return

    metrics = ["macro_f1", "accuracy", "abstention_rate"]
    modes = list(by_mode.keys())
    x = range(len(metrics))
    width = 0.8 / max(1, len(modes))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(modes):
        vals = [by_mode[mode][m] for m in metrics]
        ax.bar([xi + i * width for xi in x], vals, width, label=mode)
    ax.set_xticks([xi + width * (len(modes) - 1) / 2 for xi in x],
                  labels=["macro-F1", "accuracy", "abstention"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Retrieval: uniform vs. credibility-weighted")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "uniform_vs_weighted.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  uniform_vs_weighted -> {out.relative_to(REPO_ROOT)}")


def plot_risk_coverage(run_dir: Path) -> None:
    preds = run_dir / "predictions.jsonl"
    if not preds.exists():
        return
    has_conf = False
    with open(preds) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "confidence" in json.loads(line):
                has_conf = True
            break
    if not has_conf:
        print("  risk_coverage: predictions carry no `confidence` field yet "
              "(calibrator is Phase 7) -> skip")
        return
    print("  risk_coverage: confidence field present but plotting not wired; "
          "add once the conformal calibrator emits scores.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CRAG-FIN result figures.")
    parser.add_argument(
        "--run",
        default="experiments/runs/run_001_uniform_vs_weighted/weighted_n20",
        help="Run dir containing predictions.jsonl (for the confusion matrix).",
    )
    parser.add_argument("--csv", default="experiments/results/retriever_v1.csv")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {FIG_DIR.relative_to(REPO_ROOT)}/")
    plot_confusion(REPO_ROOT / args.run)
    plot_uniform_vs_weighted(REPO_ROOT / args.csv)
    plot_risk_coverage(REPO_ROOT / args.run)
    print("Done.")


if __name__ == "__main__":
    main()
