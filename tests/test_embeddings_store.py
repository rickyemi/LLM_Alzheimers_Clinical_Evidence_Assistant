import numpy as np
import pytest

from ad_rag.embeddings import HashingEmbedder, analyze
from ad_rag.vectorstore import VectorStore


def test_analyzer_normalises_domain_tokens():
    assert "ptau217" in analyze("plasma p-tau217 levels")
    assert "abeta" in analyze("Aβ")


def test_embeddings_are_unit_norm_and_deterministic():
    e = HashingEmbedder(512)
    a, b = e.embed(["amyloid pathology"]), e.embed(["amyloid pathology"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-5)


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder(2048)
    q, near, far = e.embed(["kidney disease raises biomarkers", "chronic kidney disease and plasma biomarkers",
                            "best pizza toppings"])
    assert q @ near > q @ far


def test_store_roundtrip_and_embedder_mismatch(settings, pipeline, tmp_path):
    pipeline.store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, expect_embedder=pipeline.embedder.name)
    assert len(loaded) == len(pipeline.store)
    assert loaded.corpus_version == pipeline.store.corpus_version
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path, expect_embedder="some-other-embedder")
