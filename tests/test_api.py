import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import get_service
from tests.conftest import DOCS, make_service


@pytest.fixture
def client():
    svc = make_service()
    app.dependency_overrides[get_service] = lambda: svc
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_stats(client):
    body = client.get("/stats").json()
    assert body["products"] == ["notifications", "payments"] and body["chunks"] > 0


def test_ingest_folder(client):
    r = client.post("/ingest", json={"product": "payments2", "source": str(DOCS / "payments")})
    assert r.status_code == 200
    assert r.json()["status"] == "done" and r.json()["documents"] == 4


def test_ingest_missing_folder_fails_cleanly(client):
    r = client.post("/ingest", json={"product": "x", "source": "/nope"})
    assert r.json()["status"] == "failed"


def test_ingest_rejects_bad_product_name(client):
    assert client.post("/ingest", json={"product": "a b", "source": "x"}).status_code == 422


def test_ask(client):
    r = client.post("/ask", json={"question": "How long are failed webhooks retried?", "product": "payments"})
    assert r.status_code == 200
    assert "3 days" in r.json()["answer"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi", "top_k": 50}).status_code == 422


def test_ask_stream_sse(client):
    with client.stream("GET", "/ask/stream", params={"question": "Can a refund be cancelled?", "product": "payments"}) as r:
        body = "".join(r.iter_text())
    assert "event: step" in body and "event: answer" in body and "Refund" in body


def test_index_serves_ui(client):
    assert "DocPilot" in client.get("/").text


def test_ingest_refuses_paths_outside_docs_root(client):
    r = client.post("/ingest", json={"product": "etc", "source": "/etc"})
    assert r.json()["status"] == "failed" and "DOCS_ROOT" in r.json()["detail"]
