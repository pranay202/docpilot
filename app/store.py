"""Vector stores: in-memory (dev/tests) and PostgreSQL + pgvector (production)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.embeddings import cosine
from app.ingest.chunker import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> int: ...
    def search(self, embedding: list[float], k: int = 5, product: str | None = None) -> list[Hit]: ...
    def products(self) -> list[str]: ...
    def count(self) -> int: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> int:
        for c in chunks:
            self._chunks[c.id] = c
        return len(chunks)

    def search(self, embedding: list[float], k: int = 5, product: str | None = None) -> list[Hit]:
        pool = [c for c in self._chunks.values() if product is None or c.product == product]
        hits = [Hit(c, cosine(embedding, c.embedding)) for c in pool]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:k]

    def products(self) -> list[str]:
        return sorted({c.product for c in self._chunks.values()})

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    def __init__(self, dsn: str, dim: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id TEXT PRIMARY KEY,
                product TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({dim}) NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS doc_chunks_product_idx ON doc_chunks (product)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx ON doc_chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )

    def upsert(self, chunks: list[Chunk]) -> int:
        from pgvector import Vector

        with self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO doc_chunks (id, product, title, section, source, text, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                [(c.id, c.product, c.title, c.section, c.source, c.text, Vector(c.embedding)) for c in chunks],
            )
        return len(chunks)

    def search(self, embedding: list[float], k: int = 5, product: str | None = None) -> list[Hit]:
        from pgvector import Vector

        sql = (
            "SELECT id, product, title, section, source, text, 1 - (embedding <=> %s) AS score "
            "FROM doc_chunks {where} ORDER BY embedding <=> %s LIMIT %s"
        )
        vec = Vector(embedding)
        if product:
            rows = self._conn.execute(sql.format(where="WHERE product = %s"), (vec, product, vec, k)).fetchall()
        else:
            rows = self._conn.execute(sql.format(where=""), (vec, vec, k)).fetchall()
        return [Hit(Chunk(r[0], r[1], r[2], r[3], r[4], r[5]), float(r[6])) for r in rows]

    def products(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT DISTINCT product FROM doc_chunks ORDER BY 1").fetchall()]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) FROM doc_chunks").fetchone()[0]
