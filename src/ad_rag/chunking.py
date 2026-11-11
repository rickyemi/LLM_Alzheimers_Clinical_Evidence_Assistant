"""Structure-aware chunking.

Why: an embedding should represent ONE focused idea. We split on markdown
headings first (so a chunk never straddles two topics), then pack paragraphs up
to `size` characters, carrying a short overlap so an answer that straddles a
boundary is still retrievable. Each chunk keeps its title + section so the
embedding (and the citation) retain context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

from .ingestion import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    section: str
    text: str
    corpus_version: str = ""

    @property
    def embed_text(self) -> str:
        """Text actually embedded: a contextual header + the chunk body."""
        return f"{self.title} | {self.section}\n{self.text}"

    def to_dict(self) -> Dict:
        return asdict(self)


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading, body) sections."""
    sections: List[Tuple[str, str]] = []
    heading, buf = "Introduction", []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _tail(text: str, n: int) -> str:
    """Last ~n chars of `text`, trimmed to start on a word boundary."""
    if n <= 0 or len(text) <= n:
        return "" if n <= 0 else text
    cut = text[-n:]
    i = cut.find(" ")
    return cut[i + 1:] if i != -1 else cut


def _window(text: str, size: int, overlap: int) -> List[str]:
    """Hard character windows for a single paragraph longer than `size`."""
    step = max(1, size - overlap)
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size].strip())
        if start + size >= len(text):
            break
        start += step
    return [o for o in out if o]


def chunk_section(body: str, size: int, overlap: int) -> List[str]:
    """Greedy paragraph packing with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(p) > size:                      # oversize paragraph: flush, then window it
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_window(p, size, overlap))
            continue
        if cur and len(cur) + 2 + len(p) > size:
            chunks.append(cur)
            tail = _tail(cur, overlap)
            cur = f"{tail}\n\n{p}" if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_documents(docs: List[Document], size: int, overlap: int,
                    corpus_version: str = "") -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        for section, body in split_sections(d.text):
            for i, text in enumerate(chunk_section(body, size, overlap)):
                cid = hashlib.sha1(f"{d.doc_id}|{section}|{i}|{text[:40]}".encode()).hexdigest()[:10]
                out.append(Chunk(cid, d.doc_id, d.title, d.source, section, text, corpus_version))
    return out
