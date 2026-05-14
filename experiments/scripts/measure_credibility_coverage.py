"""Measure how often retrieved passages have a known credibility tier.

Run weighted retrieval over the first N Fin-Fact claims against the
crag_fin_news_v0 collection. For each retrieved passage, extract the
publisher domain and look it up in the merged credibility dict. Report
the fraction with a non-default credibility (i.e. an actual tier hit).

Usage:
    python -m experiments.scripts.measure_credibility_coverage --n 200 --k 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from crag_fin.retriever.arxivmind_adapter import retrieve
from crag_fin.retriever.credibility_prior import DEFAULT_CREDIBILITY
from crag_fin.retriever.domain_extract import extract_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--finfact",
        default="data/finfact/raw/finfact.json",
        help="Path to Fin-Fact JSON",
    )
    args = parser.parse_args()

    load_dotenv()
    finfact = json.loads(Path(args.finfact).read_text())
    claims = [r["claim"] for r in finfact[: args.n]]

    total = 0
    known = 0
    domain_counter: Counter[str] = Counter()
    unknown_counter: Counter[str] = Counter()

    for claim in claims:
        passages = retrieve(claim, k=args.k, weighted=True, collection="news")
        for p in passages:
            total += 1
            domain = extract_domain(p.source) or "(unparseable)"
            domain_counter[domain] += 1
            if p.credibility != DEFAULT_CREDIBILITY:
                known += 1
            else:
                unknown_counter[domain] += 1

    coverage = known / total if total else 0.0
    print(f"Sampled {args.n} claims x k={args.k} = {total} retrievals")
    print(f"Coverage (passages with known tier): {known}/{total} = {coverage:.1%}")
    print()
    print("Top 10 domains seen:")
    for domain, count in domain_counter.most_common(10):
        print(f"  {count:5d}  {domain}")
    if unknown_counter:
        print()
        print("Top 5 unknown domains (candidates for the hand-coded dict):")
        for domain, count in unknown_counter.most_common(5):
            print(f"  {count:5d}  {domain}")


if __name__ == "__main__":
    main()
