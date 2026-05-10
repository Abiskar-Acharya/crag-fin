# CLAUDE.md — CRAG-FIN code project

This file tells Claude Code how to work on this project. It is auto-loaded at session start.

## What this project is

CRAG-FIN is a calibrated, source-credibility-weighted, agentic verifier for financial misinformation. Five components: decomposer, retriever, agentic reasoner, conformal calibrator, PRISM-style auditor. Built on top of the ArXivMind RAG substrate. See `docs/architecture.md` for the design and `README.md` for quick start.

## Sandbox boundary — where things live

| Path | What lives there | Read | Write |
|---|---|---|---|
| `~/ai-projects/crag-fin/` | All code, tests, configs, notebooks | yes | yes |
| `~/Documents/Obsidian/Noosphere/006 Projects/CRAG-FIN/` | All documentation, plans, paper notes, session notes, concepts | yes | yes |
| `~/Projects/fake_news_detection/` | CN6000 RoBERTa baseline (read for weights/data only) | yes | no |
| `~/ai-projects/glm-rag-pipeline/` | ArXivMind retrieval substrate (reuse, do not modify) | yes | no |
| Anywhere else | Off-limits unless user explicitly grants access | no | no |

**Documentation rule:** all narrative, planning, decision, and session-note content goes to the Obsidian vault path. The code project stays clean — only code, tests, configs, notebooks, and the README/architecture doc.

## Code style rules

1. **Simple over clever.** If a junior engineer can't read the function in 60 seconds, it's too clever. Rewrite.
2. **Comment what, not how.** A function comment explains *what* the function returns and *why* it exists. Inline comments only where the code looks surprising.
3. **Iterate.** Write the smallest version that works. Run it. Confirm the output is right. Then add the next layer. Do not write 200 lines and then debug.
4. **Type hints everywhere.** Functions and dataclasses use Python type hints. `mypy` must pass before commit.
5. **Tests for every new module.** When you add a function in `src/crag_fin/X/`, you add at least one test in `tests/test_X.py`. CI must pass before commit.
6. **Database / index optimisation only when measured.** Do not pre-optimise. If retrieval is slow, profile first; pick the slowest call; fix that one.
7. **No magic strings.** Configuration goes to `.env` or a YAML file. Hard-coded paths, model names, thresholds are forbidden.
8. **Reproducibility.** Set seeds. Pin dependencies. Save raw outputs as JSON before computing metrics. The pipeline must reproduce a saved result from a saved JSON without re-inferencing.

## Notebook conventions

Notebooks are sequential and runnable end-to-end. They live in `notebooks/` and follow the order in `notebooks/README.md`.

- Use markdown cells liberally. Every code cell has a markdown cell above it explaining what the cell does and what to expect.
- Output cells stay committed (so reviewers can read without running). Strip outputs only if a notebook gets too large for git.
- A notebook is "done" when it runs top-to-bottom on a clean kernel without errors and produces the expected output.

## Working session flow

At the start of every session, Claude Code does the following in order:

1. Read this `CLAUDE.md`
2. Read `~/Documents/Obsidian/Noosphere/006 Projects/CRAG-FIN/CRAG-FIN.md` (the project hub) to see status
3. Read `~/Documents/Obsidian/Noosphere/006 Projects/CRAG-FIN/08_30_day_plan.md` and identify the current day
4. Read the relevant Day X section in detail
5. Confirm with the user which sub-tasks to attempt
6. Build, iterating one sub-task at a time

At the end of every session, Claude Code writes a session note (see "Session note format" below).

## Session note format

After every working session, write a session note to `~/Documents/Obsidian/Noosphere/006 Projects/CRAG-FIN/Sessions/Session-NNN-YYYY-MM-DD.md` where NNN is zero-padded session number.

Use this template (matches the Noosphere vault Session Note Template):

```markdown
---
project: "[[CRAG-FIN]]"
session_number: NNN
date: YYYY-MM-DD
start_time: HH:MM
end_time: HH:MM
duration_min: N
phase: "Phase N — name"
day: "Day N"
author: Abiskar X Claude Code
pai_rating: TBD
---

# Session NNN — DD MMM YYYY

## Objective
What we were trying to accomplish in this session, in one sentence.

## What we did
- Bullet list of concrete actions, with [[wikilinks]] to files/notebooks created or modified
- Each bullet is a thing that shipped, not a thing we discussed

## Code changes
| Path | Action | Description |
|---|---|---|
| `src/crag_fin/X.py` | created | what it does |
| `tests/test_X.py` | created | what it tests |

## What worked
- The thing that worked, and why it worked

## What broke
- The thing that broke, what the error was, what we tried, and what fixed it (or what is still broken)

## What did not work but should
- Things that ran without erroring but produced wrong outputs
- Why the output was wrong
- The hypothesis for next session

## Decisions
- Any choices made during the session that are not in 07_PRD.md (and any that override 07_PRD.md, with reason)

## Open questions
- Things to figure out next session

## Next steps
- Concrete tasks for the next session, with date estimates
- These become Obsidian tasks: `- [ ] Task description #session #crag-fin 📅 YYYY-MM-DD`

## Rating
- Self-rated 1–10. See [[CLAUDE.md]] of vault for rating scale.
- One sentence on what made it that rating.
```

Place the rating in frontmatter `pai_rating` and inline at bottom. Both required.

## Verification before commit

Before any `git commit`, verify all of:

1. `pytest` passes
2. `ruff check .` passes
3. `mypy src/crag_fin` passes
4. The notebook(s) modified in this session run top-to-bottom on a clean kernel
5. `.env` is not staged
6. Large files (>50 MB) are not staged

If any of these fail, fix before committing. Do not commit "WIP" or "fixing tests" placeholder commits.

## Epistemic rules (non-negotiable)

When writing code or making decisions:

- **Distinguish observation, inference, and assumption.** Comments and session notes should label which is which.
- **Do not anchor on user-provided estimates.** Compute your own first, then compare.
- **Admit knowledge gaps.** If you don't know whether a library does X, say so. Do not fabricate.
- **Challenge user claims that don't match the code.** If the user says "this works" and the test fails, the test is the truth.
- **Verify before stating.** If a benchmark claim is in a paper, the paper is in `papers/` of the vault. Check the paper before quoting the number.

For factual claims in session notes, use confidence labels: `[VERIFIED]`, `[INFERRED]`, `[UNCERTAIN]`, `[SPECULATIVE]`.

## What this project is NOT (out of scope)

- Multimodal misinformation (image + text) — text only
- Continual / online learning — RAG handles temporal cutoff
- Real-time / streaming detection — batch is sufficient
- Multilingual — English-only, UK-regulated focus
- Production deployment — local Docker only for prototype

If a task pulls toward any of these, refuse the task and surface the conflict to the user.

## Where to look for context

| If you need to know about... | Read this in the vault |
|---|---|
| The atoms of the project | `01_first_principles.md` |
| Why we made each decision | `02_socratic_questions.md` + `06_socratic_review.md` |
| What the finished system looks like | `03_top_down_architecture.md` |
| What to build today | `04_bottom_up_buildplan.md` (Days 1–3) + `08_30_day_plan.md` (Days 4–30) |
| Locked tech / dataset / model choices | `07_PRD.md` |
| A specific paper | `papers/` folder |
| A specific concept | `concepts/` folder |
| Project status | `CRAG-FIN.md` |
| Past session notes | `Sessions/` folder |

If a question is not answered in the vault, ask the user before assuming.

## Communication conventions

When telling the user what shipped:

- Lead with what changed in the codebase, not what was discussed
- File paths use absolute paths or paths relative to project root
- Diffs > prose where possible
- No "I successfully implemented..." preamble; just the result

When telling the user something broke:

- Lead with the failure mode (one sentence)
- Then the error message verbatim
- Then what was tried
- Then the current state (broken / partially fixed / fixed)
- Do not bury bad news under positive framing

When asking the user a clarifying question:

- One question at a time
- Offer 2–3 options when possible, not open-ended
- Skip clarification if the answer is clear from the vault

## Git conventions

- Branch per phase: `phase-1-skeleton`, `phase-2-tooling`, etc.
- Commit messages: `<phase>(<component>): <imperative summary>` e.g. `phase-1(retriever): wire ArXivMind adapter`
- Squash-merge phase branches into `main` at end of each phase
- Tag at end of each phase: `phase-1-complete`, `phase-2-complete`, etc.
- Tag at end of 30 days: `v0.1-prototype`

## Hand-off rule

At the end of every session, the project must be in a state where another engineer (or you, returning in 2 weeks) can pick up where you left off using only:

1. `README.md`
2. The latest session note in the vault
3. `CRAG-FIN.md` hub note in the vault

If any of those would mislead the next engineer, fix them before closing the session.
