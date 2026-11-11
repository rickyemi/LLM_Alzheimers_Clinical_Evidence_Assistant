"""Offline indexing pipeline: load -> PHI gate -> chunk -> embed -> store.

Why a separate batch pipeline? Indexing is expensive and changes rarely, while
serving must be fast and stateless. The only shared state is the versioned index
artifact on disk, so the two can be deployed and scaled independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder, get_embedder
from .ingestion import Document, corpus_version, inventory, load_documents
from .phi import scan_text
from .vectorstore import VectorStore


@dataclass
class IndexReport:
    corpus_version: str
    n_docs_loaded: int
    n_docs_indexed: int
    n_chunks: int
    quarantined: List[Dict] = field(default_factory=list)   # docs blocked by the PHI gate
    inventory: List[Dict] = field(default_factory=list)


def phi_gate(docs: List[Document]):
    """Split documents into (clean, quarantined). Quarantined docs never reach the index."""
    clean, quarantined = [], []
    for d in docs:
        findings = scan_text(d.text)
        if findings:
            quarantined.append({"doc_id": d.doc_id, "kinds": sorted({f.kind for f in findings}),
                                "n_findings": len(findings)})
        else:
            clean.append(d)
    return clean, quarantined


def build_index(settings: Settings, embedder: Embedder | None = None, save: bool = True):
    """Build (and optionally persist) the index. Returns (store, embedder, report)."""
    embedder = embedder or get_embedder(settings.embedding)
    docs = load_documents(settings.path("corpus_dir"))
    clean, quarantined = phi_gate(docs)
    version = corpus_version(clean)
    chunks = chunk_documents(clean, settings.chunking.size_chars,
                             settings.chunking.overlap_chars, version)
    store = VectorStore(embedder.dim, embedder.name, version)
    if chunks:
        store.add(chunks, embedder.embed([c.embed_text for c in chunks]))
    if save:
        store.save(settings.path("index_dir"),
               extra={"chunk_size": settings.chunking.size_chars,
                      "chunk_overlap": settings.chunking.overlap_chars})
    report = IndexReport(version, len(docs), len(clean), len(chunks), quarantined, inventory(clean))
    return store, embedder, report
