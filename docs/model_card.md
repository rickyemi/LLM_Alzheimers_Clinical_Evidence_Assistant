# Model and system card: AD Clinical Evidence Assistant v0.1.0

**Status:** research and education prototype. **Not for clinical use.**

## System

| Item | Value |
|---|---|
| Type | RAG pipeline plus bounded tool-using agent |
| Default generator | Extractive baseline (quotes retrieved sentences; offline) |
| Optional generator | Anthropic Claude via `AD_RAG_LLM_BACKEND=anthropic` (model id from `ANTHROPIC_MODEL`, default `claude-sonnet-5-5`) |
| Default embedder | Hashing embedder (offline). Optional `sentence-transformers/all-MiniLM-L6-v2` |
| Index | FAISS `IndexFlatIP` or NumPy fallback, versioned manifest |
| Corpus | 10 synthetic documents (`data/raw`) |

## Intended use

Help qualified clinicians and medical affairs staff find and summarise published guidance and internal assay documents, with citations.

## Out of scope

Individual diagnosis, individual dosing or treatment advice, interpretation of a named patient's results, screening of asymptomatic people.

## Evaluation (golden set, 33 cases, offline configuration)

| Metric | Value | Gate |
|---|---|---|
| Retrieval hit@4 | 1.000 | >= 0.90 |
| MRR | 0.979 | n/a |
| Keyword recall | 0.792 | n/a |
| Faithfulness (lexical proxy) | 1.000 | >= 0.95 |
| Citation validity | 1.000 | >= 0.95 |
| Refusal accuracy | 1.000 | >= 0.95 |
| Unsafe refusal rate | 1.000 | 1.00 |

Interpretation: retrieval is strong on a tiny corpus; faithfulness is trivially 1.0 for an extractive model; keyword recall shows the baseline
quotes the wrong sentence about 1 in 5 times. Do not generalise these numbers to a real corpus or to a generative model.

## Limitations and risks

* Synthetic data; fictional assay "DemoAssay A" and illustrative cutoffs in `configs/biomarker_cutoffs.yaml`.
* Small corpus makes retrieval look easier than it is.
* Refusal threshold (0.16) sits close to the weakest in-scope score; near-miss questions may be answered or refused incorrectly.
* Lexical faithfulness proxy cannot detect subtle unsupported claims from a generative model.
* No demographic fairness evaluation.

## Monitoring and updates

Ungrounded-rate and retrieval-score drift alerts, hash-chained audit log, golden-set regression gates in CI. Update this card with every release.
