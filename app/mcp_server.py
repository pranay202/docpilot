"""MCP server exposing the documentation to AI coding agents.

Run over stdio (Claude Code, Cursor, etc.):
    python -m app.mcp_server
Or over HTTP:
    python -m app.mcp_server --http --port 8001
"""

from __future__ import annotations

import argparse
import re

from fastmcp import FastMCP

from app.schemas import AskRequest
from app.service import DocService, get_service

STEP_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+)")
SETUP_WORDS = "integrate integration set up setup install configure quickstart getting started steps"


def build_mcp(service: DocService | None = None) -> FastMCP:
    mcp = FastMCP(
        "docpilot",
        instructions="Search product documentation and get integration steps before writing integration code.",
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
    def get_integration_steps(service_name: str, product: str | None = None) -> dict:
        """Return ordered setup/integration steps for a service, taken verbatim from the docs."""
        s = svc()
        if product is None and service_name in s.store.products():
            product = service_name
        hits = s.search(f"{service_name} {SETUP_WORDS}", k=8, product=product)
        steps: list[str] = []
        sources: list[str] = []
        for h in hits:
            lines = [m.group(1).strip() for line in h.chunk.text.splitlines() if (m := STEP_LINE.match(line))]
            if lines:
                steps.extend(lines)
                sources.append(f"{h.chunk.title} > {h.chunk.section} ({h.chunk.source})")
            if len(steps) >= 12:
                break
        return {"service": service_name, "steps": steps[:12], "sources": sources}

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
