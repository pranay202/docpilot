"""MCP server exposing the documentation to AI assistants and agents.

Run over stdio (any MCP client):
    python -m app.mcp_server
Or over HTTP:
    python -m app.mcp_server --http --port 8001
"""

from __future__ import annotations

import argparse
import re

from fastmcp import FastMCP

from app.agent import _steps_answer
from app.schemas import AskRequest
from app.service import DocService, get_service

STEP_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+")


def build_mcp(service: DocService | None = None) -> FastMCP:
    mcp = FastMCP(
        "docpilot",
        instructions="Search the product documentation and fetch exact how-to steps before answering users.",
    )

    def svc() -> DocService:
        return service or get_service()

    @mcp.tool
    def list_products() -> list[str]:
        """List the products whose documentation is indexed."""
        return svc().store.products()

    @mcp.tool
    def search_docs(query: str, product: str | None = None, k: int = 5) -> list[dict]:
        """Semantic search over the documentation. Returns the best matching sections with their source."""
        k = max(1, min(k, 20))
        return [
            {
                "title": h.chunk.title,
                "section": h.chunk.section,
                "source": h.chunk.source,
                "score": round(h.score, 3),
                "text": h.chunk.text,
            }
            for h in svc().search(query, k=k, product=product)
        ]

    @mcp.tool
    def ask_docs(question: str, product: str | None = None) -> dict:
        """Answer a question from the documentation, with citations."""
        return svc().ask(AskRequest(question=question, product=product)).model_dump()

    @mcp.tool
    def get_steps(task: str, product: str | None = None) -> dict:
        """Return the ordered steps for a task (e.g. "insert a table of contents"), verbatim from the docs."""
        s = svc()
        hits = s.search(f"how to {task}", k=8, product=product)
        answer = _steps_answer(hits)
        if answer is None:
            return {"task": task, "steps": [], "sources": []}
        steps = [STEP_PREFIX.sub("", line).strip() for line in answer.answer.splitlines() if STEP_PREFIX.match(line)]
        sources = [f"{c.title} > {c.section} ({c.source})" for c in answer.citations]
        return {"task": task, "steps": steps, "sources": sources}

    return mcp


mcp = build_mcp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="http", host="0.0.0.0", port=args.port)
    else:
        mcp.run()
