"""LangGraph agent: plan -> retrieve -> answer -> verify, with retries and a safe fallback.

- plan:     rewrites the question into a search query and picks a product (LLM if available).
- retrieve: vector search, widening to all products when the filtered search is empty.
- answer:   asks the LLM for strict JSON (LLMDraft); without an LLM, answers extractively.
- verify:   validates the JSON and that every citation points at a retrieved chunk.
            Invalid drafts are retried with the validation error as feedback; after
            `max_retries` the agent falls back to an extractive, cited answer instead of failing.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.embeddings import Embedder, tokenize
from app.schemas import AskResponse, Citation, LLMDraft
from app.store import Hit, VectorStore

MIN_SCORE = 0.05
PROCEDURAL = re.compile(r"^\s*(how (do|can|should) (i|we|you)|how to|what are the steps|steps to)\b", re.I)
STEP = re.compile(r"^\s*\d+[.)]\s+.+")
NOT_FOUND = "I couldn't find this in the documentation."

PLAN_PROMPT = """You turn a user question into a documentation search query.
Known products: {products}
Return only JSON: {{"query": "<search query>", "product": "<one known product or null>"}}
Question: {question}"""

ANSWER_PROMPT = """Answer the question using only the numbered documentation blocks below.
If the blocks do not contain the answer, say so.
Return only JSON: {{"answer": "<answer>", "citations": [<block numbers used>], "confidence": <0..1>}}

{context}

Question: {question}{feedback}"""


class AgentState(TypedDict, total=False):
    question: str
    product: str | None
    top_k: int
    query: str
    hits: list[Hit]
    draft: LLMDraft | None
    attempts: int
    error: str | None
    response: AskResponse


def _extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1] if start != -1 else text)


def _citation(hit: Hit, snippet: str | None = None) -> Citation:
    c = hit.chunk
    return Citation(title=c.title, section=c.section, source=c.source, snippet=(snippet or c.text)[:300])


def extractive_answer(question: str, hits: list[Hit]) -> AskResponse:
    hits = [h for h in hits if h.score >= MIN_SCORE]
    if not hits:
        return AskResponse(answer=NOT_FOUND, citations=[], confidence=0.0, grounded=False)
    q = set(tokenize(question))
    if PROCEDURAL.match(question):
        top = hits[0]
        steps = [m.group(0).strip() for line in top.chunk.text.splitlines() if (m := STEP.match(line))]
        if len(steps) >= 2:
            return AskResponse(
                answer="\n".join(steps),
                citations=[_citation(top)],
                confidence=round(max(0.0, min(top.score, 1.0)), 3),
            )
    scored = []
    for rank, h in enumerate(hits[:2]):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text):
            sent = sent.strip(" -*#`")
            if len(sent) < 20:
                continue
            overlap = len(q & set(tokenize(sent)))
            scored.append((overlap - rank * 1.0, rank, sent, h))
    scored.sort(key=lambda s: s[0], reverse=True)
    picked = [s for s in scored[:3] if s[0] > 0] or scored[:1]
    if not picked:
        top = hits[0]
        return AskResponse(answer=top.chunk.text[:600], citations=[_citation(top)], confidence=round(min(top.score, 1.0), 3))
    picked.sort(key=lambda s: (s[1], hits[s[1]].chunk.text.find(s[2])))
    used: list[Hit] = []
    for *_, h in picked:
        if h not in used:
            used.append(h)
    return AskResponse(
        answer=" ".join(s[2] for s in picked),
        citations=[_citation(h) for h in used],
        confidence=round(max(0.0, min(hits[0].score, 1.0)), 3),
    )


def build_agent(store: VectorStore, embedder: Embedder, llm=None, max_retries: int = 2):
    def plan(state: AgentState) -> AgentState:
        question, product = state["question"], state.get("product")
        query = question
        products = store.products()
        if llm is not None:
            try:
                raw = llm.invoke(PLAN_PROMPT.format(products=", ".join(products) or "none", question=question))
                data = _extract_json(raw.content)
                query = str(data.get("query") or question)
                if not product and data.get("product") in products:
                    product = data["product"]
            except Exception:  # planning is best-effort; retrieval still works on the raw question
                query = question
        if not product:
            lowered = question.lower()
            product = next((p for p in products if p.lower() in lowered), None)
        return {"query": query, "product": product, "attempts": 0, "error": None, "draft": None}

    def retrieve(state: AgentState) -> AgentState:
        vec = embedder.embed([state["query"]])[0]
        hits = store.search(vec, k=state.get("top_k", 5), product=state.get("product"))
        if not hits and state.get("product"):
            hits = store.search(vec, k=state.get("top_k", 5))
        return {"hits": hits}

    def answer(state: AgentState) -> AgentState:
        attempts = state.get("attempts", 0) + 1
        hits = [h for h in state["hits"] if h.score >= MIN_SCORE]
        if llm is None or not hits:
            return {"attempts": attempts, "response": extractive_answer(state["question"], state["hits"])}
        context = "\n\n".join(
            f"[{i}] {h.chunk.title} > {h.chunk.section}\n{h.chunk.text}" for i, h in enumerate(hits, 1)
        )
        feedback = f"\n\nYour previous reply was rejected: {state['error']}. Return valid JSON only." if state.get("error") else ""
        try:
            raw = llm.invoke(ANSWER_PROMPT.format(context=context, question=state["question"], feedback=feedback))
            draft = LLMDraft.model_validate(_extract_json(raw.content))
            return {"attempts": attempts, "draft": draft, "error": None}
        except (ValidationError, ValueError) as exc:
            return {"attempts": attempts, "draft": None, "error": f"invalid JSON: {str(exc)[:200]}"}
        except Exception as exc:  # provider/network failure
            return {"attempts": attempts, "draft": None, "error": f"llm error: {type(exc).__name__}"}

    def verify(state: AgentState) -> AgentState:
        if state.get("response"):
            return {}
        draft = state.get("draft")
        hits = [h for h in state["hits"] if h.score >= MIN_SCORE]
        if draft is None:
            return {}
        bad = [n for n in draft.citations if not 1 <= n <= len(hits)]
        if bad:
            return {"draft": None, "error": f"citations {bad} do not match any block (1..{len(hits)})"}
        cited = list(dict.fromkeys(draft.citations))
        return {
            "response": AskResponse(
                answer=draft.answer,
                citations=[_citation(hits[n - 1]) for n in cited],
                confidence=draft.confidence,
                mode="llm",
            )
        }

    def fallback(state: AgentState) -> AgentState:
        return {"response": extractive_answer(state["question"], state["hits"])}

    def route(state: AgentState) -> str:
        if state.get("response"):
            return END
        if state.get("attempts", 0) <= max_retries:
            return "answer"
        return "fallback"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_conditional_edges("verify", route, ["answer", "fallback", END])
    graph.add_edge("fallback", END)
    return graph.compile()
