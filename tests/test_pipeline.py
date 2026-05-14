"""End-to-end smoke tests for the pipeline."""

from __future__ import annotations

from unittest.mock import patch

from crag_fin.decomposer.decompose import Proposition
from crag_fin.reasoner.single_agent import Verdict
from crag_fin.retriever.passage import Passage


def test_aggregate_any_false_is_false() -> None:
    from crag_fin.pipeline import _aggregate

    assert _aggregate(["TRUE", "FALSE", "TRUE"]) == "FALSE"


def test_aggregate_all_true_is_true() -> None:
    from crag_fin.pipeline import _aggregate

    assert _aggregate(["TRUE", "TRUE"]) == "TRUE"


def test_aggregate_mixed_uncertain_is_abstain() -> None:
    from crag_fin.pipeline import _aggregate

    assert _aggregate(["TRUE", "UNCERTAIN"]) == "ABSTAIN"


def test_pipeline_orchestration_mocked() -> None:
    """Verify the pipeline glues decompose -> retrieve -> reason correctly."""
    from crag_fin.pipeline import verify_claim

    fake_props = [
        Proposition(proposition="Apple beat earnings.", type="earnings", ticker_or_entity="AAPL"),
    ]
    fake_passages = [Passage(text="Apple beat.", source="reuters.com", score=1.0)]
    fake_verdict = Verdict(label="TRUE", justification="P1 confirms.", citations=[1])

    with patch("crag_fin.pipeline.decompose", return_value=fake_props), \
         patch("crag_fin.pipeline.retrieve", return_value=fake_passages), \
         patch("crag_fin.pipeline.reason", return_value=fake_verdict):
        cv = verify_claim("Apple beat earnings.")

    assert cv.label == "TRUE"
    assert len(cv.propositions) == 1
    assert cv.propositions[0].verdict.label == "TRUE"
