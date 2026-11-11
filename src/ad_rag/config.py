"""Typed configuration loaded from configs/config.yaml with env-var overrides.

Why a typed config?  A clinical-AI system must be reproducible: every knob that
changes behaviour (chunk size, similarity threshold, model id, tool limits)
lives in ONE reviewed file, validated by pydantic, so a release can be traced to
an exact configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    corpus_dir: str = "data/raw"
    index_dir: str = "artifacts/index"
    audit_log: str = "artifacts/audit/audit.jsonl"
    review_queue: str = "artifacts/review_queue.jsonl"
    briefs_dir: str = "artifacts/briefs"
    trials_fixture: str = "data/fixtures/trials_sample.json"
    biomarker_cutoffs: str = "configs/biomarker_cutoffs.yaml"


class EmbeddingCfg(BaseModel):
    backend: str = "hashing"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_features: int = 4096


class ChunkingCfg(BaseModel):
    size_chars: int = 900
    overlap_chars: int = 150


class RetrievalCfg(BaseModel):
    top_k: int = 4
    min_similarity: float = 0.16


class LLMCfg(BaseModel):
    backend: str = "extractive"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 800
    temperature: Optional[float] = None


class AgentCfg(BaseModel):
    max_steps: int = 6
    max_calls_per_tool: int = 2
    allow_network_tools: bool = False


class MonitoringCfg(BaseModel):
    drift_window: int = 200
    ungrounded_rate_alert: float = 0.25
    score_drop_alert: float = 0.15


class Settings(BaseModel):
    project: str = "ad-clinical-evidence-assistant"
    root: Path = Field(default_factory=Path.cwd)
    paths: Paths = Paths()
    embedding: EmbeddingCfg = EmbeddingCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    llm: LLMCfg = LLMCfg()
    agent: AgentCfg = AgentCfg()
    monitoring: MonitoringCfg = MonitoringCfg()

    def path(self, name: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else self.root / p


def find_root() -> Path:
    """Locate the repo root: $AD_RAG_HOME, else walk up from cwd, else package parent."""
    env = os.environ.get("AD_RAG_HOME")
    if env:
        return Path(env).resolve()
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "configs" / "config.yaml").exists():
            return p
    return Path(__file__).resolve().parents[2]


def load_settings(config_path: Optional[str] = None, root: Optional[Path] = None) -> Settings:
    root = Path(root) if root else find_root()
    cfg_file = Path(config_path) if config_path else root / "configs" / "config.yaml"
    raw = yaml.safe_load(cfg_file.read_text()) if cfg_file.exists() else {}
    settings = Settings(root=root, **(raw or {}))

    # Environment overrides: deployment can change behaviour without editing files.
    if v := os.environ.get("AD_RAG_EMBEDDING_BACKEND"):
        settings.embedding.backend = v
    if v := os.environ.get("AD_RAG_LLM_BACKEND"):
        settings.llm.backend = v
    if v := os.environ.get("ANTHROPIC_MODEL"):
        settings.llm.model = v
    if v := os.environ.get("AD_RAG_ALLOW_NETWORK_TOOLS"):
        settings.agent.allow_network_tools = v.lower() in {"1", "true", "yes"}
    return settings
