# Data card

| Item | Detail |
|---|---|
| `raw/*.md` | 10 **synthetic educational documents** written for this repository. They paraphrase general concepts and contain **no real patient data**, no copied guideline text, and a **fictional** assay ("DemoAssay A"). |
| `eval/golden_set.jsonl` | 33 questions: 24 in-scope, 4 out-of-scope, 5 unsafe. Synthetic. |
| `eval/agent_scenarios.jsonl` | 7 agent tasks with expected tools and escalation flag. |
| `fixtures/trials_sample.json` | 3 synthetic trial records (`SYNTH-000x`) for offline tests. |
| Licence | Synthetic documents are marked CC0-1.0 in their front matter. Code is MIT. |

## To use real documents

1. Replace or add files in `data/raw/` with the front-matter block (`title`, `source_type`, `version`, `year`, `license`; the document id is the file name).
2. Confirm licence and redistribution rights for every document.
3. Run `make index`; the PHI gate quarantines documents with findings.
4. Rebuild the golden set with clinician-written questions; recalibrate `retrieval.min_similarity` on a held-out set.
