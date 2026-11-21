import asyncio

from fastmcp import Client

from app.mcp_server import build_mcp
from tests.conftest import make_service


def call(tool: str, args: dict):
    async def go():
        async with Client(build_mcp(make_service())) as c:
            return await c.call_tool(tool, args)

    return asyncio.run(go())


def test_tools_are_listed():
    async def go():
        async with Client(build_mcp(make_service())) as c:
            return {t.name for t in await c.list_tools()}

    assert asyncio.run(go()) == {"list_products", "search_docs", "ask_docs", "get_steps"}


def test_search_docs():
    result = call("search_docs", {"query": "webhook signature", "product": "payments", "k": 3}).data
    assert len(result) == 3 and result[0]["section"].endswith("Verify signatures")


def test_ask_docs_returns_citations():
    data = call("ask_docs", {"question": "What status code means I hit the rate limit?"}).structured_content
    assert "429" in data["answer"] and data["citations"]


def test_steps_for_task():
    data = call("get_steps", {"task": "set up the notifications service", "product": "notifications"}).structured_content
    assert data["steps"][0].startswith("Enable the Notifications product")
    assert data["sources"]
