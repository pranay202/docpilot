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
import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.embeddings import Embedder, tokenize
from app.llm import describe_llm_error
from app.retrieval import hybrid_search
from app.schemas import AskResponse, Citation, LLMDraft
from app.store import Hit, VectorStore

MIN_SCORE = 0.05
PROCEDURAL = re.compile(r"\b(how (do|can|should) (i|we|you)|how to|what are the steps|steps to)\b", re.I)
STEP = re.compile(r"^\s*(\d+)[.)]\s+.+")
IMPERATIVE = re.compile(
    r"^(choose|select|click|double-click|right-click|press|open|enter|type|drag|go to|use|enable|disable|clear|"
    r"set|insert|apply|add|hold down)\b",
    re.I,
)
BULLET_STEP = re.compile(r"^\s*-\s+(.+)")
NOT_FOUND = "I couldn't find this in the documentation."
log = logging.getLogger("docpilot.agent")

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
    llm_failure: str | None  # set when the provider failed in a way retrying cannot fix
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


def _steps_in(text: str) -> list[tuple[int, str]]:
    """Numbered steps, or imperative bullet lines, as (order, line)."""
    numbered = [(int(m.group(1)), m.group(0).strip()) for line in text.splitlines() if (m := STEP.match(line))]
    if numbered:
        return numbered
    bullets = [m.group(1).strip() for line in text.splitlines() if (m := BULLET_STEP.match(line))]
    imperative = [b for b in bullets if IMPERATIVE.match(b)]
    return [(i, f"- {b}") for i, b in enumerate(imperative, 1)] if len(imperative) >= 1 else []


def _steps_answer(hits: list[Hit]) -> AskResponse | None:
    """For how-to questions: the step list of the best-ranked chunk that has one (among the top 3),
    stitched together with sibling chunks of the same section when a long list was split."""
    for h in hits[:3]:
        steps = _steps_in(h.chunk.text)
        if len(steps) < 2 and not (steps and steps[0][1].startswith("- ")):
            continue
        key = (h.chunk.source, h.chunk.section)
        siblings = [s for s in hits if (s.chunk.source, s.chunk.section) == key]
        merged: dict[str, int] = {}
        for s in siblings:
            for order, line in _steps_in(s.chunk.text):
                merged.setdefault(line, order)
        lines = [line for line, _ in sorted(merged.items(), key=lambda kv: kv[1])]
        intro = next((ln.strip() for ln in h.chunk.text.splitlines() if ln.strip() and not _steps_in(ln)), "")
        if intro.endswith(":"):
            lines.insert(0, intro)
        return AskResponse(
            answer="\n".join(lines),
            citations=[_citation(h)],
            confidence=round(max(0.0, min(h.score, 1.0)), 3),
        )
    return None


def extractive_answer(question: str, hits: list[Hit]) -> AskResponse:
    hits = [h for h in hits if h.score >= MIN_SCORE]
    if not hits:
        return AskResponse(answer=NOT_FOUND, citations=[], confidence=0.0, grounded=False)
    q = set(tokenize(question))
    procedural = bool(PROCEDURAL.search(question))
    if procedural:
        steps_answer = _steps_answer(hits)
        if steps_answer is not None:
            return steps_answer
    scored = []
    for rank, h in enumerate(hits[:2]):
        for sent in re.split(r"(?<=[.!?])\s+|\n+", h.chunk.text):
            sent = sent.strip(" -*#`")
            if len(sent) < 20:
                continue
            overlap = len(q & set(tokenize(sent)))
            if procedural and IMPERATIVE.match(sent):
                overlap += 1
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
        llm_failure = None
        if llm is not None:
            try:
                raw = llm.invoke(PLAN_PROMPT.format(products=", ".join(products) or "none", question=question))
                data = _extract_json(raw.content)
                query = str(data.get("query") or question)
                if not product and data.get("product") in products:
                    product = data["product"]
            except Exception as exc:  # planning is best-effort; retrieval still works on the raw question
                failure = describe_llm_error(exc)
                if failure is not None and failure.fatal:
                    llm_failure = failure.message
                    log.error("LLM unavailable: %s. Answering without the LLM.", failure.message)
                else:
                    reason = failure.message if failure else f"{type(exc).__name__}: {str(exc)[:200]}"
                    log.warning("LLM planning failed (%s); searching with the raw question", reason)
                query = question
        if not product:
            lowered = question.lower()
            product = next((p for p in products if p.lower() in lowered), None)
        return {
            "query": query,
            "product": product,
            "attempts": 0,
            "error": llm_failure,
            "draft": None,
            "llm_failure": llm_failure,
        }

    def retrieve(state: AgentState) -> AgentState:
        hits = hybrid_search(store, embedder, state["query"], k=state.get("top_k", 5), product=state.get("product"))
        return {"hits": hits}

    def answer(state: AgentState) -> AgentState:
        attempts = state.get("attempts", 0) + 1
        hits = [h for h in state["hits"] if h.score >= MIN_SCORE]
        if llm is None or not hits or state.get("llm_failure"):
            resp = extractive_answer(state["question"], state["hits"])
            if state.get("llm_failure"):
                resp = resp.model_copy(update={"llm_error": state["llm_failure"]})
            return {"attempts": attempts, "response": resp}
        context = "\n\n".join(
            f"[{i}] {h.chunk.title} > {h.chunk.section}\n{h.chunk.text}" for i, h in enumerate(hits, 1)
        )
        feedback = f"\n\nYour previous reply was rejected: {state['error']}. Return valid JSON only." if state.get("error") else ""
        try:
            raw = llm.invoke(ANSWER_PROMPT.format(context=context, question=state["question"], feedback=feedback))
            draft = LLMDraft.model_validate(_extract_json(raw.content))
            return {"attempts": attempts, "draft": draft, "error": None}
        except Exception as exc:
            failure = describe_llm_error(exc)
            if failure is None and isinstance(exc, (ValidationError, ValueError)):
                return {"attempts": attempts, "draft": None, "error": f"invalid JSON: {str(exc)[:200]}"}
            if failure is None:
                failure_msg, fatal = f"{type(exc).__name__}: {str(exc)[:200]}", False
            else:
                failure_msg, fatal = failure.message, failure.fatal
            if fatal:  # retrying cannot help: answer from the docs now instead of burning the retry budget
                log.error("LLM unavailable: %s. Answering without the LLM.", failure_msg)
                return {"attempts": max_retries + 1, "draft": None, "error": failure_msg, "llm_failure": failure_msg}
            log.warning("LLM answer call failed (%s); attempt %d of %d", failure_msg, attempts, max_retries + 1)
            return {"attempts": attempts, "draft": None, "error": f"llm error: {failure_msg}"}

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
        resp = extractive_answer(state["question"], state["hits"])
        reason = state.get("llm_failure") or state.get("error")
        if reason:
            resp = resp.model_copy(update={"llm_error": reason})
        return {"response": resp}

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
