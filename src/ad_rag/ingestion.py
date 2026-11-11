"""Document loading, metadata parsing and corpus versioning.

Why: RAG quality is capped by corpus quality, and every answer must be traceable
to a specific corpus snapshot. We therefore (1) keep provenance metadata for each
document and (2) compute a deterministic `corpus_version` hash that is stored in
the index manifest, returned with every answer and written to the audit log.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Document:
    doc_id: str                 # file stem, used as the citation key
    title: str
    source: str                 # file name shown to users as the citation
    text: str
    meta: Dict[str, str] = field(default_factory=dict)


def parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    """Parse a minimal `---` delimited `key: value` front-matter block."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta: Dict[str, str] = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4:].lstrip("\n")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy: only needed when PDFs are present
    return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def load_documents(corpus_dir: Path) -> List[Document]:
    """Load .md/.txt/.pdf files (sorted for determinism)."""
    docs: List[Document] = []
    for path in sorted(Path(corpus_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        raw = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        docs.append(Document(
            doc_id=path.stem,
            title=meta.get("title", path.stem.replace("_", " ").title()),
            source=path.name,
            text=body,
            meta=meta,
        ))
    return docs


def corpus_version(docs: List[Document]) -> str:
    """Deterministic 12-char hash of (doc_id, content). Changes iff the corpus changes."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.doc_id):
        h.update(d.doc_id.encode())
        h.update(hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()[:12]


def inventory(docs: List[Document]) -> List[Dict[str, str]]:
    """Source inventory table (governance artifact: what is in the corpus, and its license)."""
    return [{
        "doc_id": d.doc_id,
        "title": d.title,
        "source_type": d.meta.get("source_type", "unknown"),
        "version": d.meta.get("version", "n/a"),
        "license": d.meta.get("license", "unknown"),
        "words": str(len(d.text.split())),
    } for d in docs]
