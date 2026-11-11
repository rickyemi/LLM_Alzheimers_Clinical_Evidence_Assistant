# Evaluation report

- Embedder: `hashing-4096`  |  LLM: `extractive-baseline`  |  Corpus version: `e639851bb748`
- Cases: 33

| Metric | Value | Gate | Pass |
|---|---|---|---|
| n_cases | 33 |  |  |
| retrieval_hit_at_k | 1.000 | >= 0.9 | PASS |
| mrr | 0.979 |  |  |
| keyword_recall | 0.792 |  |  |
| faithfulness | 1.000 | >= 0.95 | PASS |
| citation_validity | 1.000 | >= 0.95 | PASS |
| refusal_accuracy | 1.000 | >= 0.95 | PASS |
| unsafe_refusal_rate | 1.000 | >= 1.0 | PASS |
| latency_p50_s | 0.001 |  |  |
| latency_p95_s | 0.001 |  |  |

## By topic (in-scope)

| Topic | n | hit@k | keyword recall |
|---|---|---|---|
| appropriate_use | 4 | 1.00 | 1.00 |
| assay | 2 | 1.00 | 1.00 |
| confounders | 2 | 1.00 | 1.00 |
| definitions | 2 | 1.00 | 0.00 |
| fairness | 1 | 1.00 | 1.00 |
| governance | 3 | 1.00 | 0.33 |
| interpretation | 2 | 1.00 | 0.50 |
| overview | 2 | 1.00 | 1.00 |
| pathway | 3 | 1.00 | 1.00 |
| therapy | 3 | 1.00 | 1.00 |
