"""Wires settings, embedder, store, LLM and agent into one service object."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from app.agent import build_agent
from app.config import Settings, load_settings
from app.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from app.ingest.chunker import chunk_document
from app.ingest.loader import load_source
from app.llm import DIMENSION_AWARE, ResolvedLLM, build_chat_model, resolve_embedding_provider, resolve_provider
from app.retrieval import hybrid_search
from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.store import Hit, InMemoryStore, PgVectorStore, VectorStore

log = logging.getLogger("docpilot")


class DocService:
    def __init__(
        self, settings: Settings, store: VectorStore, embedder: Embedder, llm=None, llm_info: dict | None = None
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        # What /stats reports: {"provider": ..., "model": ...}; never the key.
        self.llm_info = llm_info or ({"provider": "custom", "model": None} if llm is not None else None)
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
        return hybrid_search(self.store, self.embedder, query, k=k, product=product)

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


def _resolve_llm(settings: Settings) -> ResolvedLLM | None:
    return resolve_provider(settings.llm_provider, settings.llm_api_key, settings.llm_model, settings.llm_base_url)


def _build_llm(settings: Settings, resolved: ResolvedLLM | None = None):
    resolved = resolved if resolved is not None else _resolve_llm(settings)
    if resolved is None:
        log.warning(
            "No LLM configured: answers are extractive. Set an API key such as OPENAI_API_KEY, "
            "GEMINI_API_KEY or OPENROUTER_API_KEY (or LLM_PROVIDER + LLM_API_KEY) in the env or .env."
        )
        return None
    return build_chat_model(resolved, settings.llm_timeout)


def _build_embedder(settings: Settings, llm: ResolvedLLM | None) -> Embedder:
    resolved = resolve_embedding_provider(
        settings.embedding_provider, llm, settings.openai_api_key, settings.embedding_model
    )
    if resolved is None:
        return HashingEmbedder(settings.embedding_dim)
    log.info("Embeddings: %s, model %s", resolved.provider.label, resolved.model)
    return OpenAIEmbedder(
        resolved.model,
        settings.embedding_dim,
        resolved.api_key,
        resolved.base_url,
        send_dimensions=resolved.provider.id in DIMENSION_AWARE,
    )


@lru_cache
def get_service() -> DocService:
    settings = load_settings()
    resolved = _resolve_llm(settings)
    embedder = _build_embedder(settings, resolved)
    store: VectorStore = (
        PgVectorStore(settings.database_url, embedder.dim) if settings.database_url else InMemoryStore()
    )
    info = {"provider": resolved.provider.id, "model": resolved.model} if resolved else None
    return DocService(settings, store, embedder, _build_llm(settings, resolved), info)
