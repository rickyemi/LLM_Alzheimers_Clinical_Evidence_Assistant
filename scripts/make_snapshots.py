#!/usr/bin/env python
"""Render real API / agent output as terminal-style PNG snapshots for the README.

Every line shown is captured from the running system (FastAPI TestClient and the
agent), not typed by hand. Requires playwright + chromium (optional dev tool).
    python scripts/make_snapshots.py
"""
import html
import json
from pathlib import Path

from fastapi.testclient import TestClient

from ad_rag import build_pipeline, load_settings
from ad_rag.agent import build_agent
from ad_rag.api import create_app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

CSS = """body{margin:0;background:#f4f4f1}.w{width:1040px;margin:0;padding:18px 22px;background:#1b1f27;color:#d7dae0;
font:13px/1.5 'DejaVu Sans Mono',Menlo,monospace}.bar{color:#8a93a3;font-size:12px;margin-bottom:10px}
.c{color:#7ee0a5}.k{color:#e5c07b}.d{color:#8a93a3}.r{color:#ff8b8b}pre{margin:0;white-space:pre-wrap}"""


def page(title, blocks):
    body = "".join(f"<pre><span class=c>$ {html.escape(cmd)}</span>\n{html.escape(out)}</pre><br>" for cmd, out in blocks)
    return f"<style>{CSS}</style><div class=w><div class=bar>{html.escape(title)}</div>{body}</div>"


def shot(html_text, name):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1084, "height": 400}, device_scale_factor=2)
        pg.set_content(html_text)
        pg.locator(".w").screenshot(path=str(OUT / name))
        b.close()
    print("wrote", OUT / name)


def trim(d, keys):
    return json.dumps({k: d[k] for k in keys if k in d}, indent=2)


def main():
    s = load_settings()
    pipe = build_pipeline(s)
    c = TestClient(create_app(s, pipe)).__enter__()

    q = "Why do assays use two cutoffs instead of one?"
    a = c.post("/ask", json={"question": q}).json()
    blocks = [("curl localhost:8000/health", json.dumps(c.get("/health").json(), indent=2)),
              (f"curl -X POST localhost:8000/ask -d '{{\"question\": \"{q}\"}}'",
               json.dumps({"grounded": a["grounded"], "refused": a["refused"], "citations_ok": a["citations_ok"],
                           "corpus_version": a["corpus_version"],
                           "sources": [{"source": x["source"], "score": x["score"]} for x in a["sources"][:3]],
                           "answer": a["answer"][:330] + " ..."}, indent=2)),
              ("curl -X POST localhost:8000/ask -d '{\"question\": \"What dose of lecanemab should I give my father?\"}'",
               trim(c.post("/ask", json={"question": "What dose of lecanemab should I give my father?"}).json(),
                    ["grounded", "refused", "refusal_reason"]))]
    shot(page("AD Clinical Evidence Assistant | API (real responses)", blocks), "api_snapshot.png")

    task = "The plasma assay ratio was 0.45 in a symptomatic adult. How should this be interpreted, and are there recruiting Alzheimer's trials?"
    r = build_agent(s, pipe).run(task)
    lines = [f"driver: {r.name if hasattr(r, 'name') else 'offline planner'}"]
    for st in r.steps:
        lines.append(f"step {st['step']}  {st['tool']:<24} {str(st['observation'])[:120]}")
        lines.append(f"        why: {st['rationale']}")
    lines.append(f"\nstop_reason={r.stop_reason}  escalated={r.escalated}  citations_ok={r.citations_ok}")
    lines.append("\n" + r.final_answer[:520] + (" ..." if len(r.final_answer) > 520 else ""))
    bad = build_agent(s, pipe).run("What dose should I give my father?")
    blocks = [("python scripts/run_agent.py \"" + task[:60] + "...\"", "\n".join(lines)),
              ("python scripts/run_agent.py \"What dose should I give my father?\"",
               f"stop_reason={bad.stop_reason}  escalated={bad.escalated}  tools={bad.tools_used}\n{bad.final_answer[:200]}")]
    shot(page("Agentic workflow | trace of tool calls (real run)", blocks), "agent_trace_snapshot.png")


if __name__ == "__main__":
    main()
