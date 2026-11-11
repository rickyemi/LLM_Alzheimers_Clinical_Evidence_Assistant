import pytest
from fastapi.testclient import TestClient

from ad_rag.api import create_app


@pytest.fixture()
def client(settings, pipeline):
    with TestClient(create_app(settings, pipeline)) as c:
        yield c


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["chunks_indexed"] > 0


def test_ask_and_feedback_and_metrics(client):
    r = client.post("/ask", json={"question": "How does chronic kidney disease affect plasma biomarkers?"}).json()
    assert r["grounded"] and r["sources"]
    assert client.post("/feedback", json={"request_id": r["request_id"], "helpful": True}).json()["status"] == "recorded"
    assert "rag_requests_total" in client.get("/metrics").text
    assert "drift" in client.get("/metrics/summary").json()


def test_agent_endpoint(client):
    r = client.post("/agent", json={"task": "The assay ratio was 0.72 in a symptomatic adult. Is it above the cutoff?"}).json()
    assert "interpret_biomarker" in r["tools_used"]


def test_validation_and_auth(client, monkeypatch):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/ask", json={"question": "What is MCI?"}).status_code == 401
    ok = client.post("/ask", json={"question": "What is mild cognitive impairment?"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200        # probes stay open
