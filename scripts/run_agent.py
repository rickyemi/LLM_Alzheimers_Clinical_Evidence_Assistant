#!/usr/bin/env python
"""Run the agent on a task and print the tool trace.

  python scripts/run_agent.py "The assay ratio was 0.45. How should it be interpreted?"
  python scripts/run_agent.py --approve --save-brief "Is testing appropriate for ...?"
Set AD_RAG_LLM_BACKEND=anthropic (and ANTHROPIC_API_KEY) to use the Claude tool-use driver.
"""
import argparse

from ad_rag import build_pipeline, load_settings
from ad_rag.agent import build_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--approve", action="store_true", help="human approval for write tools")
    ap.add_argument("--save-brief", action="store_true")
    a = ap.parse_args()
    s = load_settings()
    p = build_pipeline(s)
    agent = build_agent(s, p, approver=lambda tool, args: a.approve)
    res = agent.run(a.task, save_brief=a.save_brief)
    print(f"Driver: {agent.name} | stop: {res.stop_reason} | escalated: {res.escalated}\n")
    for st in res.steps:
        print(f"  step {st['step']}: {st['tool']:<24} {st['observation']}\n           why: {st['rationale']}")
    print("\n" + res.final_answer)


if __name__ == "__main__":
    main()
