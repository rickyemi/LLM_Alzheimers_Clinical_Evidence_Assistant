"""Embedding backends behind one interface.

Why two backends?
  * `hashing`  : stateless, deterministic, no download, no network. Used for CI,
                 unit tests and offline demos. A strong lexical baseline.
  * `sentence-transformers`: dense semantic embeddings for production quality.
Both return L2-normalised float32 vectors, so inner product == cosine similarity.

IMPORTANT: query and documents MUST use the same embedder. The index manifest
records the embedder name and refuses to load on a mismatch.
"""
from __future__ import annotations

import re
from typing import List, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .config import EmbeddingCfg

_STOP = set("""a an and are as at be but by for from has have how i if in into is it its of on or
that the their then there these they this to was were what when where which who why will with
can do does should would could about""".split())


def _stem(tok: str) -> str:
    """Very light suffix stripping so 'biomarkers' and 'biomarker' match."""
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > 4 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def analyze(text: str) -> List[str]:
    """Lowercase, normalise domain tokens, stem, drop stopwords, add bigrams."""
    t = text.lower().replace("β", "beta").replace("ε", "e")
    t = re.sub(r"(?<=\w)-(?=\w)", "", t)          # p-tau217 -> ptau217, a-beta -> abeta
    toks = [_stem(x) for x in re.findall(r"[a-z0-9]+", t) if x not in _STOP]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Hashed bag of (stemmed) unigrams and bigrams with sublinear TF, L2-normalised."""

    def __init__(self, n_features: int = 4096):
        self.dim = n_features
        self.name = f"hashing-{n_features}"
        self._vec = HashingVectorizer(n_features=n_features, analyzer=analyze,
                                      alternate_sign=False, norm=None)

    def embed(self, texts: List[str]) -> np.ndarray:
        X = self._vec.transform(texts)
        X.data = np.log1p(X.data)                      # sublinear term frequency
        return normalize(X, norm="l2").toarray().astype("float32")


class SentenceTransformerEmbedder:
    """Dense embeddings. Lazy import so the base install stays light."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        v = self._m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalize(v, norm="l2").astype("float32")


def get_embedder(cfg: EmbeddingCfg) -> Embedder:
    if cfg.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model)
    if cfg.backend == "hashing":
        return HashingEmbedder(cfg.hashing_features)
    raise ValueError(f"Unknown embedding backend: {cfg.backend}")
