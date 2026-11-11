"""FastAPI service: the deployment surface.

Endpoints
  GET  /health           liveness + index facts (used by Docker/Kubernetes probes)
  POST /ask              grounded RAG answer with citations
  POST /agent            multi-step agentic workflow with a full tool trace
  POST /feedback         clinician thumbs up/down (feeds monitoring + improvement)
  GET  /metrics          Prometheus text format
  GET  /metrics/summary  JSON summary incl. drift status
Security: if API_KEY is set, every endpoint except /health requires header `X-API-Key`.
Privacy: raw questions are never logged; PHI is redacted before anything is stored.
"""
from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from . import __version__
from .agent import build_agent
from .config import load_settings
from .rag import build_pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AgentRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=2000)
    save_brief: bool = False
    approve_writes: bool = False        # human-in-the-loop switch for write tools


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    comment: Optional[str] = Field(None, max_length=500)


def create_app(settings=None, pipeline=None) -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or load_settings()
        p = pipeline or build_pipeline(s)
        state.update(settings=s, pipeline=p)
        yield

    app = FastAPI(title="AD Clinical Evidence Assistant", version=__version__, lifespan=lifespan,
                  description="Governed RAG + agentic AI for Alzheimer's evidence. Decision support only.")

    def require_key(x_api_key: Optional[str] = Header(None)):
        expected = os.environ.get("API_KEY")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health():
        p = state["pipeline"]
        return {"status": "ok", "version": __version__, "chunks_indexed": len(p.store),
                "corpus_version": p.store.corpus_version, "embedder": p.embedder.name, "llm": p.llm.model}

    @app.post("/ask", dependencies=[Depends(require_key)])
    def ask(req: AskRequest):
        p = state["pipeline"]
        resp = p.ask(req.question)
        return resp.to_dict()

    @app.post("/agent", dependencies=[Depends(require_key)])
    def agent(req: AgentRequest):
        s, p = state["settings"], state["pipeline"]
        a = build_agent(s, p, approver=lambda tool, args: req.approve_writes)
        t0 = time.perf_counter()
        result = a.run(req.task, save_brief=req.save_brief).to_dict()
        p.metrics.inc("agent_requests_total", escalated=str(result["escalated"]).lower())
        p.metrics.observe("agent_request_seconds", time.perf_counter() - t0)
        return result

    @app.post("/feedback", dependencies=[Depends(require_key)])
    def feedback(req: FeedbackRequest):
        p = state["pipeline"]
        p.metrics.inc("feedback_total", helpful=str(req.helpful).lower())
        if p.audit:
            p.audit.append("feedback", request_id=req.request_id, helpful=req.helpful,
                           has_comment=bool(req.comment))
        return {"status": "recorded"}

    @app.get("/metrics")
    def metrics():
        return Response(state["pipeline"].metrics.render_prometheus(), media_type="text/plain")

    @app.get("/metrics/summary", dependencies=[Depends(require_key)])
    def summary():
        p = state["pipeline"]
        return {**p.metrics.summary(), "drift": p.drift.status() if p.drift else None}

    return app


# `uvicorn ad_rag.api:app` entrypoint (lifespan builds the pipeline at startup)
app = create_app()
