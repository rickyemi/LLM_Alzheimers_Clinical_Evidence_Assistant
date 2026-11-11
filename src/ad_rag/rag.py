"""The RAG pipeline: guard -> retrieve -> threshold gate -> generate -> verify.

Request flow (each stage is separately timed and logged):
  1. Input guard      : redact PHI, refuse unsafe intents / prompt injection.
  2. Retrieval        : embed the question, fetch top-k chunks.
  3. Threshold gate   : if the best similarity is too low, refuse instead of guessing
                        (the cheapest, most effective hallucination defence).
  4. Generation       : the LLM sees ONLY the retrieved passages.
  5. Output guard     : every claim must carry a citation to a retrieved source.
  6. Audit + metrics  : redacted, hash-chained, no raw user text.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .config import Settings, load_settings
from .embeddings import Embedder, get_embedder
from .guardrails import DISCLAIMER, REFUSALS, check_input, check_output
from .indexing import build_index
from .llm import LLM, get_llm
from .monitoring import AuditLog, DriftMonitor, MetricsRegistry, sha256_text
from .vectorstore import Retrieved, VectorStore

SYSTEM_PROMPT = """You are a clinical evidence assistant for Alzheimer's disease.
Answer ONLY from the numbered context passages provided.
Rules:
- If the context does not contain the answer, say so and recommend contacting a human expert. Never invent facts.
- Cite every statement with [source: <file name>] using the source shown in the passage header.
- Do not provide individualized diagnosis, dosing or treatment advice.
- Be concise and neutral."""


def build_prompt(question: str, retrieved: List[Retrieved]) -> str:
    """Augmented prompt: numbered, source-tagged passages followed by the question."""
    blocks = [f"[Passage {r.rank} | source: {r.chunk.source} | section: {r.chunk.section}]\n{r.chunk.text}"
              for r in retrieved]
    return "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"


@dataclass
class RAGResponse:
    request_id: str
    question: str                      # REDACTED question (never raw)
    answer: str
    grounded: bool
    refused: bool
    refusal_reason: Optional[str]
    sources: List[Dict] = field(default_factory=list)
    citations_ok: bool = True
    corpus_version: str = ""
    disclaimer: str = DISCLAIMER
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, llm: LLM,
                 audit: Optional[AuditLog] = None, metrics: Optional[MetricsRegistry] = None,
                 drift: Optional[DriftMonitor] = None):
        self.settings, self.store, self.embedder, self.llm = settings, store, embedder, llm
        self.audit, self.metrics, self.drift = audit, metrics or MetricsRegistry(), drift

    # ---- stage 2 -----------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Retrieved]:
        qvec = self.embedder.embed([query])[0]
        return self.store.search(qvec, k or self.settings.retrieval.top_k)

    # ---- full request ------------------------------------------------------
    def ask(self, question: str, request_id: Optional[str] = None) -> RAGResponse:
        rid = request_id or uuid.uuid4().hex[:10]
        t0 = time.perf_counter()
        guard = check_input(question)
        safe_q = guard.redacted_question
        version = self.store.corpus_version

        def finish(resp: RAGResponse) -> RAGResponse:
            resp.metrics["total_s"] = round(time.perf_counter() - t0, 4)
            self.metrics.inc("rag_requests_total", status="refused" if resp.refused else "answered")
            self.metrics.observe("rag_request_seconds", resp.metrics["total_s"])
            if self.drift is not None and "best_score" in resp.metrics:
                self.drift.record(resp.metrics["best_score"], resp.grounded)
            if self.audit:
                self.audit.append("rag_request", request_id=rid, question=safe_q,
                                  question_hash=sha256_text(question), refused=resp.refused,
                                  reason=resp.refusal_reason, grounded=resp.grounded,
                                  sources=[s["source"] for s in resp.sources],
                                  corpus_version=version, phi_kinds=guard.phi_found)
            return resp

        # 1. input guard
        if not guard.allowed:
            return finish(RAGResponse(rid, safe_q, guard.message, False, True, guard.category,
                                      corpus_version=version, metrics={}))

        # 2. retrieval
        t1 = time.perf_counter()
        retrieved = self.retrieve(safe_q)
        retrieval_s = time.perf_counter() - t1
        best = retrieved[0].score if retrieved else 0.0
        sources = [{"source": r.chunk.source, "section": r.chunk.section,
                    "score": round(r.score, 4), "chunk_id": r.chunk.chunk_id} for r in retrieved]

        # 3. threshold gate
        if best < self.settings.retrieval.min_similarity:
            return finish(RAGResponse(rid, safe_q, REFUSALS["out_of_scope"], False, True,
                                      "out_of_scope", [], corpus_version=version,
                                      metrics={"retrieval_s": round(retrieval_s, 4),
                                               "best_score": round(best, 4)}))

        # 4. generation
        t2 = time.perf_counter()
        gen = self.llm.generate(SYSTEM_PROMPT, build_prompt(safe_q, retrieved))
        gen_s = time.perf_counter() - t2

        # 5. output guard
        og = check_output(gen.text, [r.chunk.source for r in retrieved])
        refused_by_llm = "does not contain the answer" in gen.text
        resp = RAGResponse(rid, safe_q, gen.text, grounded=not refused_by_llm and og.citation_ok,
                           refused=refused_by_llm, refusal_reason="no_answer_in_context" if refused_by_llm else None,
                           sources=sources, citations_ok=og.citation_ok or refused_by_llm,
                           corpus_version=version,
                           metrics={"retrieval_s": round(retrieval_s, 4), "generation_s": round(gen_s, 4),
                                    "best_score": round(best, 4), "input_tokens": gen.input_tokens,
                                    "output_tokens": gen.output_tokens, "model": gen.model})
        if not resp.citations_ok:
            resp.answer += "\n\n[Review needed: citation check failed. Sources retrieved are listed separately.]"
        return finish(resp)


def build_pipeline(settings: Optional[Settings] = None, rebuild: bool = False,
                   with_audit: bool = True) -> RAGPipeline:
    """Factory used by the API, scripts and notebooks. Loads the persisted index
    (building it first if absent or `rebuild=True`)."""
    settings = settings or load_settings()
    embedder = get_embedder(settings.embedding)
    index_dir = settings.path("index_dir")
    if rebuild or not (index_dir / "manifest.json").exists():
        store, embedder, _ = build_index(settings, embedder, save=True)
    else:
        store = VectorStore.load(index_dir, expect_embedder=embedder.name)
    audit = AuditLog(settings.path("audit_log")) if with_audit else None
    drift = DriftMonitor(settings.monitoring.drift_window, settings.monitoring.ungrounded_rate_alert,
                         settings.monitoring.score_drop_alert)
    return RAGPipeline(settings, store, embedder, get_llm(settings.llm), audit, MetricsRegistry(), drift)
