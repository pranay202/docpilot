from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_queues():
    r = client.post("/ingest", json={"product": "payments", "source": "./docs/payments"})
    assert r.status_code == 202
    assert r.json()["status"] == "queued"


def test_ask_validates_input():
    r = client.post("/ask", json={"question": "hi", "top_k": 50})
    assert r.status_code == 422
