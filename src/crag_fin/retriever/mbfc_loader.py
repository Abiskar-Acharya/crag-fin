"""Load MBFC-scraped tiers and merge with the hand-coded credibility dict."""

from __future__ import annotations

import csv
from pathlib import Path

from crag_fin.retriever.credibility_prior import TIER_DICT as HAND_CODED_TIERS

# MBFC's `factual_reporting` field maps cleanly to numeric credibility.
# Day-8 plan locked these values.
MBFC_TO_CREDIBILITY: dict[str, float] = {
    "VERY_HIGH": 1.0,
    "HIGH": 0.8,
    "MOSTLY_FACTUAL": 0.6,
    "MIXED": 0.4,
    "LOW": 0.2,
    "VERY_LOW": 0.1,
}


def load_mbfc_tiers(csv_path: Path) -> dict[str, float]:
    """Return {domain: credibility} from an MBFC CSV.

    Skips rows with an unrecognised `mbfc_factual_reporting` value.
    """
    out: dict[str, float] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = (row.get("domain") or "").strip().lower()
            label = (row.get("mbfc_factual_reporting") or "").strip().upper()
            if not domain or label not in MBFC_TO_CREDIBILITY:
                continue
            out[domain] = MBFC_TO_CREDIBILITY[label]
    return out


def load_credibility_dict(mbfc_csv: Path | None = None) -> dict[str, float]:
    """Merge MBFC + hand-coded credibility dicts. Hand-coded entries WIN on conflict
    because they are domain-vetted by us, not by MBFC.
    """
    if mbfc_csv is None:
        return dict(HAND_CODED_TIERS)
    merged = load_mbfc_tiers(mbfc_csv)
    merged.update(HAND_CODED_TIERS)  # hand-coded overrides MBFC on conflict
    return merged
