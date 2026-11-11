"""Shared fixtures. Everything runs offline: hashing embedder + extractive LLM."""
import pytest

from ad_rag import build_pipeline, load_settings


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("artifacts")
    s = load_settings()
    s.embedding.backend = "hashing"
    s.llm.backend = "extractive"
    s.agent.allow_network_tools = False
    # Redirect every write location to a temp dir so tests never touch the repo.
    s.paths.index_dir = str(tmp / "index")
    s.paths.audit_log = str(tmp / "audit" / "audit.jsonl")
    s.paths.review_queue = str(tmp / "review_queue.jsonl")
    s.paths.briefs_dir = str(tmp / "briefs")
    return s


@pytest.fixture(scope="session")
def pipeline(settings):
    return build_pipeline(settings, rebuild=True)
