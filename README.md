# DocPilot

An agentic documentation assistant for multi-product platforms. A **FastAPI** backend answers questions over product documentation with **RAG** and cited sources, and an **MCP server** exposes the same knowledge to AI coding agents, so they can pull the exact integration steps for a service instead of guessing.

![CI](https://github.com/pranay202/docpilot/actions/workflows/ci.yml/badge.svg)

## Features

- **Ingestion:** Markdown folders or a sitemap URL. Heading-aware chunking keeps the section path (`Payments > Webhooks > Retries`) for precise citations.
- **Vector search:** PostgreSQL + pgvector (HNSW, cosine) in production, an in-memory store for development and tests.
- **LangGraph agent:** `plan → retrieve → answer → verify`.
  - The model must return strict JSON (`answer`, `citations`, `confidence`), validated with Pydantic.
  - Invalid JSON or citations that point at no retrieved chunk are rejected and retried with the error as feedback.
  - After the retry budget, or when the LLM provider fails, the agent falls back to an extractive, cited answer instead of erroring.
- **MCP server (FastMCP):** `list_products`, `search_docs`, `ask_docs`, `get_integration_steps`, over stdio or HTTP.
- **Streaming:** `GET /ask/stream` streams each agent step and the final answer over Server-Sent Events; the bundled chat UI uses it.
- **Offline by default:** without `OPENAI_API_KEY` it uses a deterministic hashing embedder and extractive answers, so tests and evals run with no network or keys.
- **Evals in CI:** 30 Q/A pairs measure retrieval hit rate, answer accuracy and citation precision; CI fails below threshold.

## Architecture

```
 Browser (chat UI) ──SSE──▶ FastAPI ─┐
                                     ├─▶ LangGraph agent ──▶ LLM (OpenAI, optional)
 AI coding agent ──MCP──▶ FastMCP ───┘        │ plan → retrieve → answer → verify → (fallback)
                                              ▼
                                   PostgreSQL + pgvector  (doc chunks + embeddings)
```

| Path | What it does |
|---|---|
| `app/ingest/loader.py` | Markdown folder and sitemap loaders (HTML → text, nav/script stripped) |
| `app/ingest/chunker.py` | Heading-aware chunking with overlap, code-fence safe |
| `app/embeddings.py` | Hashing embedder (offline) and OpenAI embeddings |
| `app/store.py` | In-memory and pgvector stores behind one interface |
| `app/agent.py` | LangGraph agent, JSON validation, retries, extractive fallback |
| `app/service.py` | Wiring, ingestion pipeline, streaming |
| `app/main.py` | REST + SSE API and chat UI |
| `app/mcp_server.py` | MCP tools for AI coding agents |
| `evals/` | Dataset and evaluation runner |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Index the sample docs and ask a question:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"product": "payments", "source": "sample_docs/payments"}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do I verify webhook signatures?"}'
```

Open http://localhost:8000 for the chat UI.

### With PostgreSQL and an LLM

```bash
export OPENAI_API_KEY=sk-...        # optional: enables LLM answers and OpenAI embeddings
docker compose up --build           # api on :8000, MCP over HTTP on :8001, pgvector on :5433
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/stats` | Indexed chunk count, products, whether an LLM is configured |
| POST | `/ingest` | Index a Markdown folder (inside `DOCS_ROOT`) or a sitemap URL for a product |
| POST | `/ask` | Question → `{answer, citations[], confidence, mode, grounded}` |
| GET | `/ask/stream` | Same, streamed over SSE (`step` events, then `answer`) |

## MCP server

```bash
python -m app.mcp_server                 # stdio
python -m app.mcp_server --http --port 8001
```

Example client config (`.mcp.json` for MCP-capable coding agents):

```json
{
  "mcpServers": {
    "docpilot": { "command": "python", "args": ["-m", "app.mcp_server"], "cwd": "/path/to/docpilot" }
  }
}
```

| Tool | Returns |
|---|---|
| `list_products()` | Indexed products |
| `search_docs(query, product?, k?)` | Matching sections with source and score |
| `ask_docs(question, product?)` | Cited answer |
| `get_integration_steps(service_name, product?)` | Ordered setup steps taken verbatim from the docs, with sources |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | unset (in-memory) | PostgreSQL with pgvector |
| `OPENAI_API_KEY` | unset (offline mode) | Enables LLM answers and OpenAI embeddings |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `384` | Embeddings |
| `DOCS_ROOT` | working directory | Only folders inside it can be ingested |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking |
| `MAX_ANSWER_RETRIES` | `2` | Retries before the extractive fallback |

## Tests and evals

```bash
pytest -q                                   # 31 tests; the pgvector test runs when DATABASE_URL is set
python -m evals.run_eval                    # metrics on evals/dataset.jsonl
```

Current offline baseline (hashing embedder, extractive answers, 30 questions): **hit@5 1.00, answer accuracy 0.83, citation precision 0.83**. CI runs the full suite against a pgvector service container and fails the build if hit@5 < 0.9 or accuracy < 0.75.

## Roadmap

- Hybrid search (BM25 + vectors) and a reranker
- Per-tenant API keys and rate limiting
- Incremental re-ingestion by content hash
- LLM-mode eval run with a model-graded rubric

## License

MIT
