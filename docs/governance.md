# AI governance

Working document for a **non-device clinical decision-support** tool for qualified clinicians.
It is a template to adapt with your quality, privacy and regulatory teams. It is not legal advice.

## Frameworks mapped

| Framework | How this repository addresses it |
|---|---|
| NIST AI RMF (Govern, Map, Measure, Manage) | Intended use and risk class (Map), golden-set gates and fairness report (Measure), drift monitor and incident runbook (Manage), roles below (Govern) |
| FDA CDS guidance and GMLP | Transparent sources and citations, clinician stays in the loop, no individual diagnosis or dosing; a SaMD assessment is required before any scope expansion |
| HIPAA | PHI gate before indexing, redaction before logging, no PHI in prompts or traces, access control and audit |
| EU AI Act | Treat as potentially high-risk: documentation, logging, human oversight, accuracy and robustness testing |

## Roles

| Role | Responsibility |
|---|---|
| Clinical owner | Signs the intended use, golden set, and corpus changes |
| Data science lead | Model, retrieval, evaluation, drift response |
| Quality and regulatory | Change control, SaMD assessment, audit readiness |
| Privacy and security | PHI controls, access, vendor review (LLM provider, LangSmith) |
| Review board | Go or no-go at each gate |

## Controls checklist

| Control | Evidence in repo |
|---|---|
| Intended use and exclusions written down | `docs/model_card.md`, notebook 01 |
| Licensed, versioned, provenance-tracked corpus | `data/README.md`, `ingestion.inventory` |
| PHI never indexed | `phi.py`, `indexing.phi_gate`, tests |
| Unsafe intents refused before the model | `guardrails.py`, golden `unsafe` set |
| Every claim cited and verified | `guardrails.check_output` |
| Release gates block regressions | `evaluation.GATES`, `scripts/run_eval.py`, CI |
| Subgroup reporting | notebook 04 step 5 (extend with demographic strata) |
| Tamper-evident audit trail | `monitoring.AuditLog` |
| Drift detection | `monitoring.DriftMonitor`, notebook 06 |
| Human approval for writes | agent approval gate |
| Change control | any change to model, prompt, corpus, threshold, or tools re-runs gates and updates the model card |

## Change control

Re-run `make test eval notebooks` and update `docs/model_card.md` for any change to: model or model version, system prompt,
retrieval threshold, chunking, embedder, corpus, cutoffs file, or the tool set.

## Red-team practice

Keep a growing list of adversarial inputs (injection, dosing disguised as research, named-patient questions, requests in other
languages). Every incident or near miss becomes a golden-set case.

## Known gaps (be honest in reviews)

* Corpus and golden set are **synthetic**; no clinical validity is claimed.
* Threshold calibrated on 4 out-of-scope questions; needs a held-out calibration set.
* Extractive baseline has low keyword recall (0.79); a generative model has not been evaluated here.
* Regex PHI detection misses unstructured names; add NER (for example Presidio) before real data.
* Fairness reporting is by topic only; demographic strata need real labelled data.
