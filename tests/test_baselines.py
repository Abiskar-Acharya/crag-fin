"""Smoke tests for src/crag_fin/baselines/."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not Path(os.environ.get("ROBERTA_CN6000_PATH", "")).exists(),
    reason="CN6000 weights not available on this machine",
)


def test_roberta_cn6000_classifies_one_claim() -> None:
    """The baseline returns one Prediction per input claim, with score in [0, 1]."""
    from crag_fin.baselines.roberta_cn6000 import Prediction, RobertaCN6000

    model = RobertaCN6000()
    preds = model.classify(["Apple is under FCA investigation for accounting fraud."])

    assert len(preds) == 1
    assert isinstance(preds[0], Prediction)
    assert 0.0 <= preds[0].score <= 1.0
    assert preds[0].label in {"FAKE", "REAL"}
