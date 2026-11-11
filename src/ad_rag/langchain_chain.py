"""Optional LangChain (LCEL) adapter over the same index, embedder and prompt.

Why: many teams standardise on LangChain for orchestration and LangSmith for
tracing. This adapter shows the one-to-one mapping while reusing the repo's
embedder, chunks and system prompt, so results stay comparable.

    hand-rolled (rag.py)            LangChain equivalent
    --------------------            ------------------------------------
    VectorStore.search              InMemoryVectorStore.as_retriever
    build_prompt                    ChatPromptTemplate
    LLM.generate                    any chat model (e.g. ChatAnthropic) or a Runnable
    RAGPipeline.ask                 LCEL chain (the `|` pipeline)
    monitoring.py + audit           LangSmith tracing (LANGCHAIN_TRACING_V2=true)

Requires:  pip install -e ".[langchain]"
LangSmith tracing needs no code changes: set LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY
and LANGCHAIN_PROJECT in the environment.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from .embeddings import Embedder
from .llm import LLM
from .rag import SYSTEM_PROMPT
from .vectorstore import VectorStore


class AdEmbeddings(Embeddings):
    """Wrap the repo's embedder in LangChain's Embeddings interface."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed([text])[0].tolist()


def format_docs(docs: List[Document]) -> str:
    """Same numbered, source-tagged passage format as rag.build_prompt."""
    return "\n\n".join(
        f"[Passage {i} | source: {d.metadata['source']} | section: {d.metadata['section']}]\n{d.page_content}"
        for i, d in enumerate(docs, 1))


def build_langchain_rag(store: VectorStore, embedder: Embedder, llm: Optional[LLM] = None,
                        chat_model=None, k: int = 4):
    """Return an LCEL chain: question -> retrieve -> prompt -> model -> text.

    Pass `chat_model` (e.g. langchain_anthropic.ChatAnthropic) for a real LLM, or
    `llm` (the repo's LLM interface, e.g. ExtractiveLLM) for offline runs.
    """
    docs = [Document(page_content=c.text, metadata={"source": c.source, "section": c.section})
            for c in store.chunks]
    vs = InMemoryVectorStore.from_documents(docs, AdEmbeddings(embedder))
    retriever = vs.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ])
    if chat_model is not None:
        model = chat_model
    elif llm is not None:
        model = RunnableLambda(lambda pv: llm.generate(SYSTEM_PROMPT, pv.to_string()).text)
    else:
        raise ValueError("Provide either chat_model or llm")
    return ({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser())
