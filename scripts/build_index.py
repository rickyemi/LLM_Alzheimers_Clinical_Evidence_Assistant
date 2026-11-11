#!/usr/bin/env python
"""Build the versioned vector index from data/raw (PHI gate -> chunk -> embed -> save)."""
import json

from ad_rag import load_settings
from ad_rag.indexing import build_index


def main():
    s = load_settings()
    store, embedder, rep = build_index(s, save=True)
    print(f"Embedder        : {embedder.name}")
    print(f"Corpus version  : {rep.corpus_version}")
    print(f"Docs indexed    : {rep.n_docs_indexed}/{rep.n_docs_loaded}")
    print(f"Chunks          : {rep.n_chunks}")
    if rep.quarantined:
        print("QUARANTINED (PHI findings):", json.dumps(rep.quarantined, indent=2))
    print(f"Index written to: {s.path('index_dir')}")


if __name__ == "__main__":
    main()
