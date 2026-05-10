"""CN6000 RoBERTa fake-news baseline.

Loads the Year-1 CN6000 checkpoint trained on ISOT (RobertaForSequenceClassification
with num_labels=1, sigmoid output for binary fake/real). Reused here to provide
the cross-domain failure hook on Fin-Fact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


@dataclass(frozen=True)
class Prediction:
    """One prediction from the CN6000 baseline."""

    claim: str
    score: float  # sigmoid probability in [0, 1]
    label: str  # "FAKE" if score >= 0.5 else "REAL" (CN6000 convention)


class RobertaCN6000:
    """Wrapper around the CN6000 RoBERTa checkpoint.

    Loads `roberta-base` and overrides its weights from a saved
    `state = {'state_dict': ..., 'optimizer_dict': ..., 'bestscore': ...}` dict.
    """

    LABEL_THRESHOLD = 0.5
    MAX_LENGTH = 512

    def __init__(self, weights_path: str | Path | None = None, device: str | None = None):
        path = Path(weights_path or os.environ["ROBERTA_CN6000_PATH"])
        if not path.exists():
            raise FileNotFoundError(f"CN6000 weights not found at {path}")

        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=1
        )

        state = torch.load(path, map_location=self.device, weights_only=False)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.model.load_state_dict(sd)
        self.model.to(self.device)  # type: ignore[arg-type]
        # Put module into inference mode (equivalent to nn.Module.eval())
        self.model.train(False)

    def classify(self, claims: list[str]) -> list[Prediction]:
        """Score each claim. Higher score = more fake-news-like (CN6000 convention)."""
        if not claims:
            return []

        enc = self.tokenizer(
            claims,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        return [
            Prediction(
                claim=claim,
                score=float(p),
                label="FAKE" if p >= self.LABEL_THRESHOLD else "REAL",
            )
            for claim, p in zip(claims, probs, strict=False)
        ]
