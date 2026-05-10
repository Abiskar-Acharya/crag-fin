# Data Provenance

Every dataset in this folder must have a row in this README documenting its source, license, retrieval date, and processing.

## Datasets

### finfact/

- **Source:** Rangapur, Wang, and Shu (2023), "Fin-Fact: A Benchmark Dataset for Multimodal Financial Fact Checking and Explanation Generation"
- **arXiv:** [2309.08793](https://arxiv.org/abs/2309.08793)
- **License:** Check the upstream repo
- **Retrieval date:** TBD
- **Processing:** Use FinFact data only (text); ignore image fields
- **Splits:** train 1562 / val 391 / test 1304

### finguard/

- **Source:** Carlos GMartin, "Financial Truth Guard"
- **GitHub:** https://github.com/carlos-gmartin/Financial-Truth-Guard
- **License:** Check the upstream repo
- **Retrieval date:** TBD
- **Splits:** Built locally — train 2900 / val 600 / test 1500 (per FMDLlama paper)

### source_credibility/

- **Source (preferred):** NewsGuard (paid)
- **Source (fallback):** Media Bias / Fact Check (free) https://mediabiasfactcheck.com
- **License:** Check before redistributing — store locally only
- **Retrieval date:** TBD
- **Schema:** `domain, tier, factual_reporting, bias_label, last_checked`

### uk_fca/

- **Source:** Financial Conduct Authority enforcement notices, https://www.fca.org.uk/news/news-stories
- **License:** Public sector content; check FCA terms before redistribution
- **Retrieval date:** TBD
- **Schema:** `notice_id, date, firm_name, summary, full_text, outcome`

### adversarial/

- **Source:** LLM-generated financial misinformation produced under ethics-approved protocol
- **License:** NOT FOR REDISTRIBUTION — synthetic only; for stress test under M5
- **Retrieval date:** TBD (Year 3, post-ethics)
- **Note:** This is a Year-3 chapter, not in the 30-day plan.

## Rules

1. Anything in this folder that is large (>100 MB) is gitignored. See root `.gitignore`.
2. Re-runnable download scripts live in `../scripts/setup_data.py`. Do not commit raw downloads.
3. Every processing step must be a script. No manual cleaning. If you do clean by hand, write a script that reproduces it.
