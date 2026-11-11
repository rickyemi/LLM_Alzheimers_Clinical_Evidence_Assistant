"""Agent drivers: a deterministic offline planner and a Claude tool-use loop.

WHERE AGENTIC AI COMES IN
  RAG answers ONE question from ONE retrieval. An agent handles tasks that need
  several different capabilities in sequence: search the evidence, run a
  deterministic calculation, check a policy, look up trials, then decide whether
  to answer or hand off to a human. The LLM is the planner; the tools do the
  work; the guardrails bound the autonomy.

TWO DRIVERS, ONE SET OF CONTROLS
  * ClaudeToolAgent : production driver. Claude decides which tool to call next
                      (Anthropic Messages API tool use) in a bounded loop.
  * OfflineAgent    : deterministic keyword planner. A transparent stand-in for the
                      LLM planner so tests, CI and demos run with no API key. It is
                      NOT a claim of LLM-level planning quality.
  Both route every tool call through ToolRegistry (allowlist, schema validation,
  call caps, human approval for writes, hash-chained audit) and finish with the
  same output guard.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import Settings
from ..guardrails import DISCLAIMER, REFUSALS, check_input, check_output
from ..rag import SYSTEM_PROMPT, RAGPipeline
from .tools import ToolContext, ToolRegistry, default_tools

AGENT_SYSTEM = """You are a clinical evidence agent for Alzheimer's disease supporting qualified clinicians.
You may call tools to gather evidence. Follow these rules strictly:
1. Ground every factual statement in a tool result and cite it as [source: <file name>].
2. Use interpret_biomarker and check_appropriate_use for any numeric interpretation or eligibility logic. Never do clinical arithmetic yourself.
3. Never give individualized diagnosis, dosing or treatment advice. If asked, call escalate_to_human and explain you cannot.
4. If the knowledge base lacks the answer, say so and call escalate_to_human. Never guess.
5. Never include patient identifiers. Use scan_for_phi if text may contain any.
6. Stop as soon as you can answer. Keep the final answer concise and structured."""


@dataclass
class AgentResult:
    task: str
    final_answer: str
    steps: List[Dict] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    escalated: bool = False
    stop_reason: str = "completed"
    citations_ok: bool = True
    sources: List[str] = field(default_factory=list)
    tokens: Dict[str, int] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> Dict:
        return asdict(self)


def _brief(tool: str, out: Dict) -> str:
    """One-line observation summary for the trace."""
    if not out.get("ok"):
        return f"error: {out.get('error')}"
    if tool == "search_knowledge_base":
        return f"{len(out['passages'])} passages, best_score={out['best_score']}, grounded={out['grounded']}"
    if tool == "interpret_biomarker":
        return f"band={out['band']}"
    if tool == "check_appropriate_use":
        return f"eligible={out['eligible']}, mode={out['use_mode']}"
    if tool == "search_clinical_trials":
        return f"{len(out['trials'])} trials ({out['source']})"
    if tool == "escalate_to_human":
        return f"ticket={out['ticket']}"
    if tool == "scan_for_phi":
        return f"phi_found={out['phi_found']}"
    return "ok"


# --------------------------------------------------------------------------- #
# Offline deterministic planner
# --------------------------------------------------------------------------- #
class OfflineAgent:
    name = "offline-keyword-planner"

    def __init__(self, registry: ToolRegistry, pipeline: RAGPipeline, settings: Settings):
        self.reg, self.pipeline, self.settings = registry, pipeline, settings

    # -- planning ------------------------------------------------------------
    @staticmethod
    def _focus_query(task: str) -> str:
        """Rewrite the task into a retrieval query by dropping numerals (values, ages).
        Numbers carry no topical signal for the embedding and only dilute similarity;
        they are handled by deterministic tools instead. An LLM planner does this
        implicitly when it writes the tool arguments."""
        return re.sub(r"\s+", " ", re.sub(r"\d[\d.,]*(-year-old|-year|%)?", " ", task)).strip() or task

    def _plan(self, task: str) -> List[Tuple[str, Dict, str]]:
        t = task.lower()
        plan: List[Tuple[str, Dict, str]] = [
            ("search_knowledge_base", {"query": self._focus_query(task), "k": 4},
             "Gather cited evidence first; every later step builds on it."),
        ]
        m = re.search(r"(?:ratio|value|result|level|score)\D{0,15}(\d*\.\d+|\d+)", t)
        if m and re.search(r"p-?tau|ratio|assay|biomarker", t):
            plan.append(("interpret_biomarker", {"value": float(m.group(1))},
                         "A numeric value was given: compare it with the configured cutoffs deterministically."))
        if re.search(r"\b(eligible|appropriate|screen\w*|should .{0,25}(test|tested))\b", t):
            age = re.search(r"(\d{2})[- ]?(?:year|yr|y/o)", t)
            symptomatic = bool(re.search(r"symptom|memory loss|cognitive complaint|forgetful|decline", t)) \
                and not re.search(r"asymptomatic|no symptoms|without symptoms", t)
            setting = "specialty" if re.search(r"memory clinic|specialist|neurolog|specialty", t) else "primary"
            evaluated = bool(re.search(r"evaluat|work-?up|assessed|examined", t))
            args = {"cognitive_symptoms": symptomatic, "setting": setting, "clinical_evaluation_done": evaluated}
            if age:
                args["age"] = int(age.group(1))
            plan.append(("check_appropriate_use", args,
                         "The question asks about eligibility: apply the appropriate-use rules."))
        if re.search(r"\b(trial|trials|recruiting|enroll\w*)\b", t):
            plan.append(("search_clinical_trials", {"status": "RECRUITING", "max_results": 3},
                         "The user asked about trials: query the registry."))
        return plan[: self.settings.agent.max_steps]

    # -- run -----------------------------------------------------------------
    def run(self, task: str, save_brief: bool = False) -> AgentResult:
        guard = check_input(task)
        steps: List[Dict] = []
        if not guard.allowed:
            out = self.reg.execute("escalate_to_human", {"reason": f"guardrail:{guard.category}",
                                                         "summary": guard.redacted_question})
            steps.append({"step": 1, "rationale": "Guardrail blocked an unsafe request; hand off to a human.",
                          "tool": "escalate_to_human", "ok": out["ok"], "observation": _brief("escalate_to_human", out)})
            return AgentResult(guard.redacted_question, guard.message, steps, ["escalate_to_human"],
                               True, "guardrail_refusal")
        safe = guard.redacted_question
        results: Dict[str, Dict] = {}
        for i, (tool, args, why) in enumerate(self._plan(safe), 1):
            out = self.reg.execute(tool, args)
            results[tool] = out
            steps.append({"step": len(steps) + 1, "rationale": why, "tool": tool, "args": args,
                          "ok": out["ok"], "observation": _brief(tool, out)})

        kb = results.get("search_knowledge_base", {})
        if not kb.get("ok") or not kb.get("grounded"):
            out = self.reg.execute("escalate_to_human",
                                   {"reason": "no grounded evidence", "summary": safe})
            steps.append({"step": len(steps) + 1, "rationale": "Evidence too weak: escalate instead of guessing.",
                          "tool": "escalate_to_human", "ok": out["ok"], "observation": _brief("escalate_to_human", out)})
            return AgentResult(safe, REFUSALS["out_of_scope"], steps, [s["tool"] for s in steps],
                               True, "no_grounded_evidence")

        answer = self._synthesize(safe, results)
        if save_brief:
            out = self.reg.execute("save_evidence_brief", {"title": "Evidence brief", "content": answer})
            steps.append({"step": len(steps) + 1, "rationale": "Persist the brief (write action: needs approval).",
                          "tool": "save_evidence_brief", "ok": out["ok"], "observation": _brief("save_evidence_brief", out)})
        sources = sorted({p["source"] for p in kb["passages"]})
        og = check_output(answer, sources)
        return AgentResult(safe, answer, steps, [s["tool"] for s in steps], False, "completed",
                           og.citation_ok, sources)

    # -- synthesis -----------------------------------------------------------
    def _synthesize(self, task: str, results: Dict[str, Dict]) -> str:
        kb = results["search_knowledge_base"]
        prompt = "CONTEXT:\n" + "\n\n".join(
            f"[Passage {i} | source: {p['source']} | section: {p['section']}]\n{p['text']}"
            for i, p in enumerate(kb["passages"], 1)) + f"\n\nQUESTION: {task}"
        evidence = self.pipeline.llm.generate(SYSTEM_PROMPT, prompt).text
        parts = ["**Evidence from the knowledge base**", evidence]
        if (b := results.get("interpret_biomarker")) and b.get("ok"):
            parts += ["", "**Biomarker interpretation (deterministic)**",
                      f"- Value {b['value']} on `{b['assay']}` is in the **{b['band'].replace('_', ' ')}** band "
                      f"(cutoffs {b['lower_cutoff']} / {b['upper_cutoff']}). {b['interpretation']} _{b['disclaimer']}_"]
        if (a := results.get("check_appropriate_use")) and a.get("ok"):
            parts += ["", "**Appropriate-use check (rule based)**",
                      f"- Eligible: **{a['eligible']}** (mode: {a['use_mode']})"]
            parts += [f"- {r}" for r in a["reasons"] + a["warnings"]]
        if (t := results.get("search_clinical_trials")) and t.get("ok"):
            parts += ["", f"**Clinical trials** (source: {t['source']})"]
            parts += [f"- {x['nct_id']}: {x['title']} ({x['status']})" for x in t["trials"]] or ["- none found"]
        parts += ["", f"_{DISCLAIMER}_"]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Claude tool-use driver
# --------------------------------------------------------------------------- #
class ClaudeToolAgent:
    name = "claude-tool-use"

    def __init__(self, registry: ToolRegistry, settings: Settings, client=None):
        self.reg, self.settings = registry, settings
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def run(self, task: str, save_brief: bool = False) -> AgentResult:
        guard = check_input(task)
        steps: List[Dict] = []
        if not guard.allowed:
            out = self.reg.execute("escalate_to_human", {"reason": f"guardrail:{guard.category}",
                                                         "summary": guard.redacted_question})
            steps.append({"step": 1, "rationale": "Guardrail refusal", "tool": "escalate_to_human",
                          "ok": out["ok"], "observation": _brief("escalate_to_human", out)})
            return AgentResult(guard.redacted_question, guard.message, steps, ["escalate_to_human"],
                               True, "guardrail_refusal")

        messages: List[Dict[str, Any]] = [{"role": "user", "content": guard.redacted_question}]
        seen_sources: set = set()
        tokens = {"input": 0, "output": 0}
        final, stop = "", "max_steps"
        for _ in range(self.settings.agent.max_steps):
            kwargs = dict(model=self.settings.llm.model, max_tokens=self.settings.llm.max_tokens,
                          system=AGENT_SYSTEM, tools=self.reg.schemas(), messages=messages)
            if self.settings.llm.temperature is not None:
                kwargs["temperature"] = self.settings.llm.temperature
            resp = self.client.messages.create(**kwargs)
            tokens["input"] += resp.usage.input_tokens
            tokens["output"] += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                final = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                stop = "completed"
                break
            tool_results = []
            for b in resp.content:
                if getattr(b, "type", "") != "tool_use":
                    continue
                out = self.reg.execute(b.name, dict(b.input))
                if b.name == "search_knowledge_base" and out.get("ok"):
                    seen_sources |= {p["source"] for p in out["passages"]}
                steps.append({"step": len(steps) + 1, "rationale": "chosen by model", "tool": b.name,
                              "args": dict(b.input), "ok": out["ok"], "observation": _brief(b.name, out)})
                tool_results.append({"type": "tool_result", "tool_use_id": b.id,
                                     "content": json.dumps(out)[:8000], "is_error": not out["ok"]})
            messages.append({"role": "user", "content": tool_results})

        escalated = "escalate_to_human" in [s["tool"] for s in steps]
        if stop == "max_steps":
            out = self.reg.execute("escalate_to_human", {"reason": "agent hit step cap", "summary": guard.redacted_question})
            escalated = True
            final = final or "I could not finish within the step limit; the request was sent to a human reviewer."
        og = check_output(final, sorted(seen_sources))
        citations_ok = og.citation_ok or not seen_sources or escalated
        if not citations_ok:
            final += "\n\n[Review needed: citation check failed.]"
        return AgentResult(guard.redacted_question, final + f"\n\n_{DISCLAIMER}_", steps,
                           [s["tool"] for s in steps], escalated, stop, citations_ok, sorted(seen_sources), tokens)


def build_agent(settings: Settings, pipeline: RAGPipeline,
                approver: Optional[Callable[[str, Dict], bool]] = None, client=None):
    """Factory: Claude driver when llm.backend == 'anthropic', else the offline planner."""
    ctx = ToolContext(settings, pipeline, approver or (lambda tool, args: False))
    reg = ToolRegistry(default_tools(), ctx, settings.agent.max_calls_per_tool)
    if settings.llm.backend == "anthropic":
        return ClaudeToolAgent(reg, settings, client)
    return OfflineAgent(reg, pipeline, settings)
