# DocPilot

An agentic help-centre assistant for a brand with several products. A **FastAPI** backend answers user questions over the official documentation with **RAG** and cited sources. An **MCP server** gives AI assistants and agents the same knowledge, including exact how-to steps for a task.

It is built and evaluated on a real, well-known brand: **LibreOffice**, across four products: **Writer, Calc, Impress and Draw**. That is 291 official help guides and 923 indexed sections.

![CI](https://github.com/pranay202/docpilot/actions/workflows/ci.yml/badge.svg)

## Results on LibreOffice help

50 hand-written user questions (`evals/libreoffice.jsonl`): 15 Writer, 15 Calc, 12 Impress, 8 Draw. Each question names the help page and section that answers it and the exact phrase the answer must contain. All of them were checked against the pinned help text.

| Metric | Offline mode (no API key) |
|---|---|
| Retrieval hit@5 (right section in top 5) | **0.92** |
| Answer accuracy (expected phrase in answer) | **0.78** |
| Citation precision (cited sections that are the right one) | **0.69** |
| Average latency per question | ~23 ms |

Offline mode uses a deterministic hashing embedder and extractive answers, so these numbers are a floor that needs no API key or network. CI recomputes them on every push and fails the build if hit@5 < 0.85 or accuracy < 0.70.

Example (Writer):

```
Q: How do I add line numbers to the whole document?
A: 1. Choose Tools - Line Numbering.
   2. Select Show numbering, and then select the options that you want.
   3. Click OK.
   [Adding Line Numbers > To Add Line Numbers to an Entire Document — writer/numbering_lines.xhp]
```

## Features

- **Ingestion:** LibreOffice help XML (`.xhp`), Markdown/MDX folders, or a sitemap URL.
  - Heading-aware chunking keeps the section path (`Page Numbers > To Start With a Defined Page Number`) for precise citations.
  - The `.xhp` parser keeps ordered steps, tips and tables, takes the default (non-macOS) shortcut variants, and drops index bookmarks, comments and images.
- **Hybrid retrieval:** vector recall (6× candidates), then a lexical re-rank that rewards query terms in the section heading. Help guides put the task in the heading ("To Insert a Table of Contents"), which makes this a strong, cheap signal.
- **Vector store:** PostgreSQL + pgvector (HNSW, cosine) in production, in-memory for development and tests.
- **LangGraph agent:** `plan → retrieve → answer → verify`.
  - The model must return strict JSON (`answer`, `citations`, `confidence`), validated with Pydantic.
  - Invalid JSON, or citations that point at no retrieved section, are rejected and retried with the error as feedback.
  - After the retry budget, or when the LLM provider fails, the agent falls back to an extractive, cited answer instead of erroring.
  - For "how do I…" questions it returns the documented steps in order, stitching a step list back together when chunking split it.
- **MCP server (FastMCP):** `list_products`, `search_docs`, `ask_docs`, `get_steps`, over stdio or HTTP.
- **Streaming:** `GET /ask/stream` streams each agent step and the final answer over Server-Sent Events; the bundled chat UI uses it.

## Architecture

```
 Browser (chat UI) ──SSE──▶ FastAPI ─┐
                                     ├─▶ LangGraph agent ──▶ LLM (any provider, optional)
 AI assistant ──MCP──▶ FastMCP ──────┘        │ plan → retrieve → answer → verify → (fallback)
                                              ▼
                     hybrid retrieval (vector recall + heading-aware re-rank)
                                              ▼
                                   PostgreSQL + pgvector  (help sections + embeddings)
```

| Path | What it does |
|---|---|
| `app/ingest/xhp.py` | LibreOffice help XML → Markdown |
| `app/ingest/loader.py` | Folder loader (`.xhp`, `.md`, `.mdx`) and sitemap loader |
| `app/ingest/chunker.py` | Heading-aware chunking with overlap, code-fence safe |
| `app/retrieval.py` | Hybrid search: vector recall + lexical re-rank |
| `app/llm.py` | Provider registry: picks the LLM and embeddings endpoint from env/.env |
| `app/embeddings.py` | Hashing embedder (offline) and OpenAI-compatible embeddings |
| `app/store.py` | In-memory and pgvector stores behind one interface |
| `app/agent.py` | LangGraph agent, JSON validation, retries, step extraction, extractive fallback |
| `app/service.py` | Wiring, ingestion pipeline, streaming |
| `app/main.py` | REST + SSE API and chat UI |
| `app/mcp_server.py` | MCP tools |
| `evals/` | LibreOffice and sample datasets, evaluation runner |
| `scripts/fetch_libreoffice_help.sh` | Fetches the help guides at a pinned commit |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
./scripts/fetch_libreoffice_help.sh          # Writer, Calc, Impress, Draw guides → brand_docs/libreoffice
uvicorn app.main:app --reload
```

Index the four products and ask a question:

```bash
for p in writer calc impress draw; do
  curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' \
    -d "{\"product\": \"$p\", \"source\": \"brand_docs/libreoffice/$p\"}"
done
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How do I refresh a pivot table after the source data changes?", "product": "calc"}'
```

Open http://localhost:8000 for the chat UI. To index the four products automatically at startup, run `PRELOAD_DOCS=brand_docs/libreoffice uvicorn app.main:app` (Docker Compose sets this).

### With an LLM (Gemini, OpenAI, OpenRouter, Anthropic, ...)

Put one API key in `.env` (copy `.env.example`). DocPilot picks the provider from whichever key is set:

```bash
cp .env.example .env
echo 'GEMINI_API_KEY=AIza...' >> .env      # or OPENAI_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY / ...
uvicorn app.main:app                       # the log line "LLM: Google Gemini, model ..." confirms it
curl localhost:8000/stats                  # {"llm": true, "llm_provider": "gemini", ...}
```

| `LLM_PROVIDER` | Key env var | Default model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-2.5-flash` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `together` | `TOGETHER_API_KEY` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `ollama` / `lmstudio` | none (local) | `llama3.1` / `local-model` |
| `custom` | `LLM_API_KEY` + `LLM_BASE_URL` | set `LLM_MODEL` |

All of them speak the OpenAI chat API, so one client (`langchain-openai`) serves every provider and no extra SDK is installed. An OpenRouter or Gemini key pasted into `OPENAI_API_KEY` by mistake is recognised by its prefix and routed to the right provider.

### With PostgreSQL

```bash
docker compose up --build           # api on :8000, MCP over HTTP on :8001, pgvector on :5433; reads .env
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/stats` | Indexed section count, products, LLM provider and model (or `llm: false`) |
| POST | `/ingest` | Index a docs folder (inside `DOCS_ROOT`) or a sitemap URL for a product |
| POST | `/ask` | Question → `{answer, citations[], confidence, mode, grounded}` |
| GET | `/ask/stream` | Same, streamed over SSE (`step` events, then `answer`) |

## MCP server

```bash
python -m app.mcp_server                 # stdio
python -m app.mcp_server --http --port 8001
```

Example client config (`.mcp.json`):

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
| `get_steps(task, product?)` | Ordered steps for a task, verbatim from the docs, with sources |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | unset (in-memory) | PostgreSQL with pgvector |
| `LLM_PROVIDER` | `auto` | `auto` (first key found), a provider id from the table above, or `none` |
| `LLM_API_KEY` | unset | Generic key for the named provider (else its own key env var) |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, ... | unset (offline mode) | Provider keys; any one enables LLM answers |
| `LLM_MODEL` | provider default | Chat model |
| `LLM_BASE_URL` | provider default | Any OpenAI-compatible `/v1` URL |
| `LLM_TIMEOUT` | `30` | Seconds per LLM call |
| `EMBEDDING_PROVIDER` | `auto` | `auto` (OpenAI if `OPENAI_API_KEY` is set, else hashing), `hashing`, `openai`, `gemini`, `mistral`, `ollama` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | provider default / `384` | Embeddings; re-ingest after changing them |
| `LOG_LEVEL` | `INFO` | Provider errors are logged as warnings before the extractive fallback |
| `DOCS_ROOT` | working directory | Only folders inside it can be ingested |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunking |
| `MAX_ANSWER_RETRIES` | `2` | Retries before the extractive fallback |

## Tests and evals

```bash
pytest -q                                                       # 38 tests
python -m evals.run_eval --dataset evals/libreoffice.jsonl --docs brand_docs/libreoffice
python -m evals.run_eval                                        # small bundled sample set
```

The pgvector test runs when `DATABASE_URL` is set, and the LibreOffice threshold test runs once the help docs are fetched. CI does both: it runs against a pgvector service container and the pinned help docs.

Eval rows name the expected help page and section, the phrase the answer must contain, and any equally correct alternative the docs give (for example *Insert - Page Numbers* as well as *Insert - Field - Page Number*).

## Data and licensing

The LibreOffice help content is © The Document Foundation and contributors, under the Mozilla Public License 2.0. It is not stored in this repository: `scripts/fetch_libreoffice_help.sh` downloads it at commit `72e318c` of [LibreOffice/help](https://github.com/LibreOffice/help), so every eval run uses the same text. LibreOffice is a trademark of The Document Foundation; this project is independent and not affiliated with it.

## Roadmap

- LLM-mode eval with a model-graded rubric
- Cross-encoder re-ranker
- Incremental re-ingestion by content hash
- Per-tenant API keys and rate limiting

## License

MIT (code). Documentation content: see *Data and licensing*.
