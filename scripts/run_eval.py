#!/usr/bin/env python
"""Run the golden-set evaluation, write reports/, and exit non-zero if a release gate fails."""
import json
import sys

from ad_rag import build_pipeline, load_settings
from ad_rag.evaluation import GATES, check_gates, evaluate, load_golden


def main():
    s = load_settings()
    p = build_pipeline(s, rebuild=True, with_audit=False)
    res = evaluate(p, load_golden(s.root / "data/eval/golden_set.jsonl"))
    gates = check_gates(res["summary"], GATES)
    out = s.root / "reports"
    out.mkdir(exist_ok=True)
    payload = {"embedder": p.embedder.name, "llm": p.llm.model, "corpus_version": p.store.corpus_version,
               "n_chunks": len(p.store), **res, "gates": gates}
    (out / "eval_results.json").write_text(json.dumps(payload, indent=2))

    lines = ["# Evaluation report", "",
             f"- Embedder: `{p.embedder.name}`  |  LLM: `{p.llm.model}`  |  Corpus version: `{p.store.corpus_version}`",
             f"- Cases: {res['summary']['n_cases']}", "", "| Metric | Value | Gate | Pass |", "|---|---|---|---|"]
    for k, v in res["summary"].items():
        g = gates.get(k)
        val = f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} | {'>= ' + str(g['threshold']) if g else ''} | "
                     f"{('PASS' if g['passed'] else 'FAIL') if g else ''} |")
    lines += ["", "## By topic (in-scope)", "", "| Topic | n | hit@k | keyword recall |", "|---|---|---|---|"]
    for t, d in res["by_topic"].items():
        kw = "n/a" if d["keyword_recall"] is None else f"{d['keyword_recall']:.2f}"
        lines.append(f"| {t} | {d['n']} | {d['hit']:.2f} | {kw} |")
    (out / "eval_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    failed = [k for k, v in gates.items() if not v["passed"]]
    if failed:
        print(f"\nRELEASE GATE FAILED: {failed}")
        sys.exit(1)
    print("\nAll release gates passed.")


if __name__ == "__main__":
    main()
