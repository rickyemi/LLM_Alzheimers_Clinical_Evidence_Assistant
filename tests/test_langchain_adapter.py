import pytest

pytest.importorskip("langchain_core")

from ad_rag.langchain_chain import build_langchain_rag  # noqa: E402
from ad_rag.llm import ExtractiveLLM  # noqa: E402


def test_lcel_chain_answers_with_citation(pipeline):
    chain = build_langchain_rag(pipeline.store, pipeline.embedder, llm=ExtractiveLLM())
    out = chain.invoke("Why do assays use two cutoffs instead of one?")
    assert "[source: two_cutoff_interpretation.md]" in out
