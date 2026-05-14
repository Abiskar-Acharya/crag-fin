"""Unit tests for the Fin-Fact metric computation."""

from __future__ import annotations

import json
from pathlib import Path


def _write_predictions(tmp_path: Path, pairs: list[tuple[str, str]]) -> Path:
    out = tmp_path / "predictions.jsonl"
    with open(out, "w") as f:
        for gold, pred in pairs:
            f.write(json.dumps({"gold": gold, "pred": pred}) + "\n")
    return out


def test_perfect_agreement(tmp_path: Path) -> None:
    from crag_fin.eval.metrics import compute_finfact_metrics

    pairs = [("true", "TRUE"), ("false", "FALSE"), ("NEI", "ABSTAIN")]
    m = compute_finfact_metrics(_write_predictions(tmp_path, pairs))
    assert m["macro_f1"] == 1.0
    assert m["accuracy"] == 1.0


def test_all_wrong(tmp_path: Path) -> None:
    from crag_fin.eval.metrics import compute_finfact_metrics

    pairs = [("true", "FALSE"), ("false", "ABSTAIN"), ("NEI", "TRUE")]
    m = compute_finfact_metrics(_write_predictions(tmp_path, pairs))
    assert m["macro_f1"] == 0.0
    assert m["accuracy"] == 0.0


def test_mixed_with_abstention(tmp_path: Path) -> None:
    from crag_fin.eval.metrics import compute_finfact_metrics

    pairs = [
        ("true", "TRUE"),
        ("true", "TRUE"),
        ("false", "FALSE"),
        ("NEI", "ABSTAIN"),
        ("true", "ABSTAIN"),
    ]
    m = compute_finfact_metrics(_write_predictions(tmp_path, pairs))
    assert m["accuracy"] == 4 / 5
    assert m["abstention_rate"] == 2 / 5
    assert 0.0 < m["macro_f1"] < 1.0


def test_unknown_labels_dropped(tmp_path: Path) -> None:
    from crag_fin.eval.metrics import compute_finfact_metrics

    pairs = [("true", "TRUE"), ("BOGUS", "FALSE"), ("false", "WAT")]
    m = compute_finfact_metrics(_write_predictions(tmp_path, pairs))
    # Only the first row survives the filter
    assert m["n"] == 1
