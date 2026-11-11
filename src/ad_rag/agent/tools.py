"""Agent tools: typed, validated, risk-classified, audited.

Design principles (see docs/agentic_ai.md):
  * The agent never runs free-form code. It can only call tools in this registry.
  * Every tool has a pydantic argument schema (validated BEFORE execution) and a
    risk class: `read` tools are free to call; `write` tools need human approval.
  * Clinical arithmetic (cutoff comparison, eligibility rules) is done by
    deterministic code, NOT by the LLM, so it is testable and reproducible.
  * Every call (allowed or denied) is appended to the hash-chained audit log.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..phi import redact, scan_text
from ..rag import RAGPipeline


# --------------------------------------------------------------------------- #
# Context + registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolContext:
    settings: Settings
    pipeline: RAGPipeline
    approver: Callable[[str, Dict], bool] = lambda tool, args: False   # default: DENY write tools


@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    func: Callable[[ToolContext, BaseModel], Dict]
    risk: str = "read"                       # read | write
    requires_approval: bool = False

    def anthropic_schema(self) -> Dict:
        """Schema in the format the Anthropic Messages API expects for tool use."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


class ToolRegistry:
    def __init__(self, tools: List[Tool], ctx: ToolContext, max_calls_per_tool: int = 2):
        self.tools = {t.name: t for t in tools}
        self.ctx = ctx
        self.max_calls = max_calls_per_tool
        self.calls: Dict[str, int] = {}

    def schemas(self) -> List[Dict]:
        return [t.anthropic_schema() for t in self.tools.values()]

    def execute(self, name: str, raw_args: Dict[str, Any]) -> Dict:
        """Run a tool through every control: allowlist, call cap, validation, approval, audit."""
        audit = self.ctx.pipeline.audit

        def log(outcome: str, **kw):
            if audit:
                audit.append("tool_call", tool=name, outcome=outcome, **kw)

        tool = self.tools.get(name)
        if tool is None:
            log("denied_unknown_tool")
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        if self.calls.get(name, 0) >= self.max_calls:
            log("denied_call_cap")
            return {"ok": False, "error": f"Call cap reached for '{name}'."}
        try:
            args = tool.args_model(**raw_args)
        except ValidationError as e:
            log("denied_invalid_args", errors=len(e.errors()))
            return {"ok": False, "error": f"Invalid arguments: {e.errors()[0]['msg']}"}
        if tool.requires_approval and not self.ctx.approver(name, args.model_dump()):
            log("denied_not_approved")
            return {"ok": False, "error": "Human approval required and not granted."}
        self.calls[name] = self.calls.get(name, 0) + 1
        t0 = time.perf_counter()
        try:
            result = tool.func(self.ctx, args)
            log("ok", seconds=round(time.perf_counter() - t0, 4))
            return {"ok": True, **result}
        except Exception as e:  # tool failures must never crash the agent loop
            log("error", error=type(e).__name__)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------- #
# Tool 1: knowledge-base search (the RAG retriever exposed as a tool)
# --------------------------------------------------------------------------- #
class SearchKBArgs(BaseModel):
    query: str = Field(..., min_length=3, description="Natural-language search query")
    k: int = Field(4, ge=1, le=8, description="Number of passages to return")


def search_knowledge_base(ctx: ToolContext, a: SearchKBArgs) -> Dict:
    retrieved = ctx.pipeline.retrieve(redact(a.query), a.k)
    best = retrieved[0].score if retrieved else 0.0
    return {
        "best_score": round(best, 4),
        "grounded": best >= ctx.settings.retrieval.min_similarity,
        "passages": [{"source": r.chunk.source, "section": r.chunk.section,
                      "score": round(r.score, 4), "text": r.chunk.text} for r in retrieved],
    }


# --------------------------------------------------------------------------- #
# Tool 2: biomarker interpretation (deterministic, config driven)
# --------------------------------------------------------------------------- #
class InterpretArgs(BaseModel):
    assay: str = Field("demo_ptau217_ratio", description="Assay key from configs/biomarker_cutoffs.yaml")
    value: float = Field(..., ge=0, description="Measured assay value")


def interpret_biomarker(ctx: ToolContext, a: InterpretArgs) -> Dict:
    cfg = yaml.safe_load(ctx.settings.path("biomarker_cutoffs").read_text())
    if a.assay not in cfg["assays"]:
        raise ValueError(f"Unknown assay '{a.assay}'. Known: {sorted(cfg['assays'])}")
    c = cfg["assays"][a.assay]
    lo, hi = c["lower_cutoff"], c["upper_cutoff"]
    if a.value < lo:
        band, note = "below_lower_cutoff", "Biomarker-negative band for this assay."
    elif a.value >= hi:
        band, note = "above_upper_cutoff", "Biomarker-positive band; clinical correlation still required."
    else:
        band, note = "intermediate", "Intermediate zone: confirmatory CSF or PET testing is suggested."
    return {"assay": a.assay, "value": a.value, "lower_cutoff": lo, "upper_cutoff": hi,
            "band": band, "interpretation": note, "disclaimer": cfg["disclaimer"]}


# --------------------------------------------------------------------------- #
# Tool 3: appropriate-use checklist (deterministic rules from the policy docs)
# --------------------------------------------------------------------------- #
class AUCArgs(BaseModel):
    cognitive_symptoms: bool = Field(..., description="Does the patient have cognitive symptoms?")
    setting: str = Field("primary", description="primary | specialty | community")
    clinical_evaluation_done: bool = Field(False, description="Has a qualified clinician evaluated the patient?")
    age: Optional[int] = Field(None, ge=0, le=120)


def check_appropriate_use(ctx: ToolContext, a: AUCArgs) -> Dict:
    reasons: List[str] = []
    warnings: List[str] = []
    eligible = True
    if a.age is not None and a.age < 18:
        eligible = False
        reasons.append("Blood biomarker testing is intended for adults.")
    if not a.cognitive_symptoms:
        eligible = False
        reasons.append("No cognitive symptoms: asymptomatic screening is not supported by the appropriate-use criteria.")
    if not a.clinical_evaluation_done:
        eligible = False
        reasons.append("A clinical evaluation by a qualified clinician should precede biomarker testing.")
    if eligible:
        mode = "triage" if a.setting in {"primary", "community"} else "triage_or_confirmatory"
        warnings.append("A positive result requires clinical correlation and, in primary care, specialist confirmation."
                        if mode == "triage" else
                        "Confirmatory use requires an assay meeting the performance thresholds in the policy.")
        reasons.append("Symptomatic adult with clinical evaluation: testing may be appropriate.")
    else:
        mode = "not_recommended"
    return {"eligible": eligible, "use_mode": mode, "reasons": reasons, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Tool 4: clinical-trial search (ClinicalTrials.gov API v2, offline fixture fallback)
# --------------------------------------------------------------------------- #
class TrialsArgs(BaseModel):
    condition: str = Field("Alzheimer Disease")
    status: str = Field("RECRUITING", description="e.g. RECRUITING, COMPLETED")
    max_results: int = Field(3, ge=1, le=10)


def search_clinical_trials(ctx: ToolContext, a: TrialsArgs) -> Dict:
    if ctx.settings.agent.allow_network_tools:
        import httpx
        r = httpx.get("https://clinicaltrials.gov/api/v2/studies",
                      params={"query.cond": a.condition, "filter.overallStatus": a.status,
                              "pageSize": a.max_results}, timeout=10)
        r.raise_for_status()
        trials = [{"nct_id": s["protocolSection"]["identificationModule"]["nctId"],
                   "title": s["protocolSection"]["identificationModule"].get("briefTitle", ""),
                   "status": s["protocolSection"]["statusModule"].get("overallStatus", "")}
                  for s in r.json().get("studies", [])]
        return {"source": "clinicaltrials.gov_api_v2", "trials": trials}
    fixture = json.loads(ctx.settings.path("trials_fixture").read_text())
    trials = [t for t in fixture["trials"] if t["status"] == a.status][: a.max_results]
    return {"source": "offline_fixture (synthetic)", "trials": trials}


# --------------------------------------------------------------------------- #
# Tool 5: PHI scan
# --------------------------------------------------------------------------- #
class PHIArgs(BaseModel):
    text: str


def scan_for_phi(ctx: ToolContext, a: PHIArgs) -> Dict:
    kinds = sorted({f.kind for f in scan_text(a.text)})
    return {"phi_found": bool(kinds), "kinds": kinds, "redacted_text": redact(a.text)}


# --------------------------------------------------------------------------- #
# Tool 6: escalate to a human (the safety valve)
# --------------------------------------------------------------------------- #
class EscalateArgs(BaseModel):
    reason: str = Field(..., min_length=3)
    summary: str = Field("", description="Short, PHI-free summary of the request")


def escalate_to_human(ctx: ToolContext, a: EscalateArgs) -> Dict:
    q = ctx.settings.path("review_queue")
    q.parent.mkdir(parents=True, exist_ok=True)
    ticket = f"REV-{uuid.uuid4().hex[:8]}"
    with open(q, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ticket": ticket, "reason": redact(a.reason),
                            "summary": redact(a.summary), "ts": time.time()}) + "\n")
    return {"ticket": ticket, "status": "queued_for_human_review"}


# --------------------------------------------------------------------------- #
# Tool 7: save an evidence brief (WRITE action, needs human approval)
# --------------------------------------------------------------------------- #
class BriefArgs(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    content: str = Field(..., min_length=10)


def save_evidence_brief(ctx: ToolContext, a: BriefArgs) -> Dict:
    d = ctx.settings.path("briefs_dir")
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in a.title)[:60]
    path = d / f"{safe}.md"
    path.write_text(f"# {a.title}\n\n{redact(a.content)}\n")
    return {"saved_to": str(path)}


def default_tools() -> List[Tool]:
    return [
        Tool("search_knowledge_base", "Search the curated Alzheimer's evidence corpus. Returns cited passages.",
             SearchKBArgs, search_knowledge_base),
        Tool("interpret_biomarker", "Compare a blood biomarker value with the configured two-cutoff bands.",
             InterpretArgs, interpret_biomarker),
        Tool("check_appropriate_use", "Check whether biomarker testing fits the appropriate-use criteria.",
             AUCArgs, check_appropriate_use),
        Tool("search_clinical_trials", "Look up Alzheimer's clinical trials (ClinicalTrials.gov).",
             TrialsArgs, search_clinical_trials),
        Tool("scan_for_phi", "Detect and redact protected health information in text.",
             PHIArgs, scan_for_phi),
        Tool("escalate_to_human", "Queue the request for human expert review when unsure or unsafe.",
             EscalateArgs, escalate_to_human),
        Tool("save_evidence_brief", "Save a markdown evidence brief to disk. Requires human approval.",
             BriefArgs, save_evidence_brief, risk="write", requires_approval=True),
    ]
