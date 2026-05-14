"""Fin-Fact metric computation.

Fin-Fact gold labels are lowercase `{true, false, NEI}`. Our pipeline emits
`{TRUE, FALSE, ABSTAIN}`. We map `true -> TRUE`, `false -> FALSE`,
`NEI -> ABSTAIN`. Macro-F1 over the 3 classes is our headline metric;
abstention rate is computed separately for diagnostic purposes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

GOLD_TO_PRED: dict[str, str] = {
    "true": "TRUE",
    "false": "FALSE",
    "NEI": "ABSTAIN",
    "nei": "ABSTAIN",
}
LABELS = ("TRUE", "FALSE", "ABSTAIN")


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_finfact_metrics(predictions_jsonl: Path) -> dict:
    """Compute macro-F1, per-class P/R/F1, and abstention rate.

    Each line of predictions_jsonl is `{"gold": "true|false|NEI", "pred": "TRUE|FALSE|ABSTAIN"}`.
    """
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
            if gold not in LABELS or pred not in LABELS:
                continue
            rows.append((gold, pred))

    if not rows:
        return {"n": 0, "macro_f1": 0.0, "per_class": {}, "abstention_rate": 0.0}

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(1 for g, p in rows if g == label and p == label)
        fp = sum(1 for g, p in rows if g != label and p == label)
        fn = sum(1 for g, p in rows if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _f1(tp, fp, fn)
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    abstention_rate = sum(1 for _, p in rows if p == "ABSTAIN") / len(rows)
    accuracy = sum(1 for g, p in rows if g == p) / len(rows)
    return {
        "n": len(rows),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "abstention_rate": abstention_rate,
        "per_class": per_class,
        "pred_dist": dict(Counter(p for _, p in rows)),
        "gold_dist": dict(Counter(g for g, _ in rows)),
    }
