# DocPilot

An agentic documentation assistant: a **FastAPI** backend that answers questions over product documentation with **RAG**, and exposes the same knowledge to AI coding agents through an **MCP server**.

> Status: in progress. The FastAPI skeleton, schemas and tests are in place; the milestones below track what is left.

## Why

Developer docs for multi-product platforms are hard to search, and AI coding agents can't read them reliably. DocPilot gives humans a chat interface with cited answers, and gives agents a structured MCP tool they can call directly.

## Architecture

```
            ┌──────────────┐        ┌─────────────────────┐
 Browser ──▶│ Next.js chat │──SSE──▶│  FastAPI  /ask      │
            └──────────────┘        │  LangGraph agent    │──▶ LLM API (OpenAI / Anthropic / local)
                                    │   plan → retrieve → │
 AI coding agent ──MCP (stdio/SSE)─▶│   answer → verify   │
                                    └─────────┬───────────┘
                                              │
                                    ┌─────────▼───────────┐
                                    │ PostgreSQL+pgvector │  doc chunks + embeddings
                                    └─────────────────────┘
```

## Stack

Python 3.14 · FastAPI · Pydantic v2 · LangGraph · FastMCP · PostgreSQL + pgvector · Next.js · Docker Compose · pytest · GitHub Actions

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| POST | `/ingest` | Index a docs source (Markdown folder or sitemap) for a product |
| POST | `/ask` | Question → structured answer with citations (`AskResponse`) |
| GET | `/ask/stream` | Same, streamed over SSE |

Answers are validated against a Pydantic schema (`answer`, `citations[]`, `confidence`), so the agent must return structured JSON or retry.

## Milestones

- [x] FastAPI skeleton, Pydantic schemas, tests, Dockerfile
- [ ] Ingestion: Markdown/sitemap loader, chunking, embeddings into pgvector
- [ ] Retrieval with per-product filters and source citations
- [ ] LangGraph agent: plan → retrieve → answer → self-check, with retries and graceful fallback on tool/LLM errors
- [ ] MCP server (FastMCP) exposing `search_docs` and `get_integration_steps` tools for coding agents
- [ ] SSE streaming endpoint
- [ ] Eval set: 30 Q/A pairs, answer accuracy + citation precision, run in CI
- [ ] Next.js chat UI
- [ ] Docker Compose (api + postgres) and GitHub Actions (lint, test, eval)

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
```

## License

MIT
