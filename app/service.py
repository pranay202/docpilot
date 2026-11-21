"""Wires settings, embedder, store, LLM and agent into one service object."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from app.agent import build_agent
from app.config import Settings, load_settings
from app.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from app.ingest.chunker import chunk_document
from app.ingest.loader import load_source
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.store import Hit, InMemoryStore, PgVectorStore, VectorStore


class DocService:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder, llm=None) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.agent = build_agent(store, embedder, llm, settings.max_answer_retries)

    def _check_source(self, source: str) -> None:
        if source.startswith(("http://", "https://")):
            return
        root = Path(self.settings.docs_root).resolve()
        path = Path(source)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_relative_to(root):
            raise PermissionError(f"Folder must be inside DOCS_ROOT ({root})")

    def ingest(self, req: IngestRequest) -> IngestResponse:
        try:
            self._check_source(req.source)
            docs = load_source(req.source, req.product)
        except Exception as exc:
            return IngestResponse(product=req.product, source=req.source, status="failed", detail=str(exc)[:300])
        chunks = [c for d in docs for c in chunk_document(d, self.settings.chunk_size, self.settings.chunk_overlap)]
        for i in range(0, len(chunks), 64):
            batch = chunks[i : i + 64]
            for c, vec in zip(batch, self.embedder.embed([f"{c.title} {c.section}\n{c.text}" for c in batch])):
                c.embedding = vec
        self.store.upsert(chunks)
        return IngestResponse(product=req.product, source=req.source, status="done", documents=len(docs), chunks=len(chunks))

    def search(self, query: str, k: int = 5, product: str | None = None) -> list[Hit]:
        return self.store.search(self.embedder.embed([query])[0], k=k, product=product)

    def ask(self, req: AskRequest) -> AskResponse:
        state = self.agent.invoke({"question": req.question, "product": req.product, "top_k": req.top_k})
        return state["response"]

    def ask_stream(self, req: AskRequest) -> Iterator[tuple[str, dict]]:
        """Yields (event, data) pairs: one 'step' per graph node, then the final 'answer'."""
        final: AskResponse | None = None
        inputs = {"question": req.question, "product": req.product, "top_k": req.top_k}
        for update in self.agent.stream(inputs, stream_mode="updates"):
            for node, delta in update.items():
                delta = delta or {}
                info: dict = {"node": node}
                if node == "retrieve":
                    info["sources"] = [f"{h.chunk.title} > {h.chunk.section}" for h in delta.get("hits", [])]
                if delta.get("error"):
                    info["error"] = delta["error"]
                yield "step", info
                if delta.get("response"):
                    final = delta["response"]
        if final is not None:
            yield "answer", final.model_dump()


def _build_llm(settings: Settings):
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_model, temperature=0, timeout=30, max_retries=1)


@lru_cache
def get_service() -> DocService:
    settings = load_settings()
    if settings.openai_api_key:
        embedder: Embedder = OpenAIEmbedder(settings.embedding_model, settings.embedding_dim)
    else:
        embedder = HashingEmbedder(settings.embedding_dim)
    store: VectorStore = (
        PgVectorStore(settings.database_url, embedder.dim) if settings.database_url else InMemoryStore()
    )
    return DocService(settings, store, embedder, _build_llm(settings))
