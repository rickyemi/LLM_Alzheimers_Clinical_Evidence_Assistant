"""Persistent vector store: FAISS when available, NumPy otherwise.

Why: retrieval must be fast and the index must be a versioned, reloadable
artifact. We persist vectors (.npy), chunk metadata (.jsonl) and a manifest that
pins the corpus version, embedder and build time. FAISS IndexFlatIP gives exact
cosine search (vectors are normalised); swap to HNSW or pgvector at scale
without touching callers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .chunking import Chunk

try:  # FAISS is optional; NumPy fallback keeps the package installable anywhere
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


@dataclass
class Retrieved:
    chunk: Chunk
    score: float          # cosine similarity in [-1, 1]
    rank: int


class VectorStore:
    def __init__(self, dim: int, embedder_name: str = "", corpus_version: str = ""):
        self.dim = dim
        self.embedder_name = embedder_name
        self.corpus_version = corpus_version
        self.chunks: List[Chunk] = []
        self.vectors = np.empty((0, dim), dtype="float32")
        self._index = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "chunks and vectors must align"
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])
        self._index = None                      # invalidate; rebuilt lazily

    def _ensure_index(self):
        if faiss is not None and self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self.vectors)

    def search(self, qvec: np.ndarray, k: int) -> List[Retrieved]:
        if not len(self.chunks):
            return []
        k = min(k, len(self.chunks))
        q = qvec.reshape(1, -1).astype("float32")
        if faiss is not None:
            self._ensure_index()
            scores, ids = self._index.search(q, k)
            scores, ids = scores[0], ids[0]
        else:
            sims = (self.vectors @ q.T).ravel()
            ids = np.argsort(-sims)[:k]
            scores = sims[ids]
        return [Retrieved(self.chunks[i], float(s), r + 1)
                for r, (i, s) in enumerate(zip(ids, scores)) if i != -1]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path, extra: Optional[dict] = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        manifest = {
            "corpus_version": self.corpus_version,
            "embedder": self.embedder_name,
            "dim": self.dim,
            "n_chunks": len(self.chunks),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "faiss": faiss is not None,
            **(extra or {}),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: Path, expect_embedder: Optional[str] = None) -> "VectorStore":
        d = Path(directory)
        man = json.loads((d / "manifest.json").read_text())
        if expect_embedder and man["embedder"] != expect_embedder:
            raise ValueError(
                f"Index was built with embedder '{man['embedder']}' but the service is "
                f"configured for '{expect_embedder}'. Rebuild the index."
            )
        store = cls(man["dim"], man["embedder"], man["corpus_version"])
        store.vectors = np.load(d / "vectors.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            store.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return store

    @staticmethod
    def read_manifest(directory: Path) -> dict:
        return json.loads((Path(directory) / "manifest.json").read_text())
