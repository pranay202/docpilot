"""Runs only when DATABASE_URL points at PostgreSQL with pgvector (docker compose up db)."""

import os
import uuid

import pytest

from app.embeddings import HashingEmbedder
from app.ingest.chunker import Chunk

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_pgvector_roundtrip():
    from app.store import PgVectorStore

    store = PgVectorStore(os.environ["DATABASE_URL"], 64)
    store._conn.execute("DROP TABLE IF EXISTS doc_chunks")
    store = PgVectorStore(os.environ["DATABASE_URL"], 64)
    e = HashingEmbedder(64)
    texts = ["verify webhook signatures with hmac", "refunds take five days", "quiet hours hold sms"]
    chunks = [Chunk(uuid.uuid4().hex, "p", "T", f"S{i}", "s.md", t, e.embed([t])[0]) for i, t in enumerate(texts)]
    assert store.upsert(chunks) == 3
    hits = store.search(e.embed(["webhook signature"])[0], k=2, product="p")
    assert hits[0].chunk.text == texts[0]
    assert store.count() == 3 and store.products() == ["p"]
    store._conn.execute("DROP TABLE doc_chunks")
