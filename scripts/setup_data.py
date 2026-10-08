#!/usr/bin/env python3
"""One-shot, idempotent data setup for CRAG-FIN.

Running this on a fresh clone does three things, each skipped if already done:

  1. Fetch the Fin-Fact benchmark into ``data/finfact/raw/`` (git clone).
  2. Seed the Fin-Fact evidence + financial-news ChromaDB indexes the
     retriever reads from (unless ``--no-index``).
  3. Report whether the external ArXivMind retrieval substrate is configured
     (``ARXIVMIND_REPO``), which the retriever needs at query time.

It intentionally does NOT try to hide the external dependency: retrieval reuses
the ArXivMind ``HybridRetriever``, so that repo must be present and pointed to
by ``ARXIVMIND_REPO`` in your ``.env`` before ``verify_claim`` will retrieve.

Usage:
    python scripts/setup_data.py            # fetch data + seed indexes
    python scripts/setup_data.py --no-index # fetch data only (fast)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINFACT_DIR = REPO_ROOT / "data" / "finfact" / "raw"
FINFACT_JSON = FINFACT_DIR / "finfact.json"
FINFACT_GIT_URL = "https://github.com/IIT-DM/Fin-Fact.git"


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def fetch_finfact() -> None:
    print("[1/3] Fin-Fact benchmark")
    if FINFACT_JSON.exists():
        size_mb = FINFACT_JSON.stat().st_size / 1e6
        print(f"  already present ({size_mb:.0f} MB) -> skip")
        return
    FINFACT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {FINFACT_GIT_URL} -> {FINFACT_DIR}")
    code = _run(["git", "clone", "--depth", "1", FINFACT_GIT_URL, str(FINFACT_DIR)])
    if code != 0 or not FINFACT_JSON.exists():
        sys.exit(
            "  ERROR: Fin-Fact clone failed or finfact.json missing.\n"
            f"  Clone it manually into {FINFACT_DIR} and re-run."
        )
    print("  done")


def seed_indexes() -> None:
    print("[2/3] ChromaDB indexes (Fin-Fact evidence + financial news)")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    for label, module in [
        ("Fin-Fact evidence", "crag_fin.tools.finfact_evidence_seed"),
        ("financial news", "crag_fin.tools.news_seed"),
    ]:
        print(f"  seeding {label} ({module}) ...")
        code = subprocess.call([sys.executable, "-m", module], cwd=str(REPO_ROOT), env=env)
        if code != 0:
            print(f"  WARNING: seeding {label} exited with code {code}. "
                  "Check your .env chroma paths and that sentence-transformers is installed.")
    print("  done")


def report_arxivmind() -> None:
    print("[3/3] ArXivMind retrieval substrate (required at query time)")
    repo = os.environ.get("ARXIVMIND_REPO", "").strip()
    if not repo:
        print("  ARXIVMIND_REPO is not set. Set it in .env to the local path of the")
        print("  glm-rag-pipeline repo (https://github.com/Abiskar-Acharya/glm-rag-pipeline).")
        return
    if Path(repo).exists():
        print(f"  found: {repo}")
    else:
        print(f"  WARNING: ARXIVMIND_REPO points to a missing path: {repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up CRAG-FIN data and indexes.")
    parser.add_argument("--no-index", action="store_true", help="Fetch data only; skip seeding.")
    args = parser.parse_args()

    fetch_finfact()
    if args.no_index:
        print("[2/3] skipped (--no-index)")
    else:
        seed_indexes()
    report_arxivmind()
    print("\nSetup complete. Next: `python -m crag_fin.pipeline \"<a claim>\"`")


if __name__ == "__main__":
    main()
