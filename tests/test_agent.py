import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agent import NOT_FOUND
from app.schemas import AskRequest
from tests.conftest import make_service

PLAN = json.dumps({"query": "verify webhook signature hmac", "product": "payments"})


def test_extractive_answer_without_llm(service):
    resp = service.ask(AskRequest(question="Which header carries the webhook signature?", product="payments"))
    assert resp.mode == "extractive"
    assert "X-AcmePay-Signature" in resp.answer
    assert resp.citations and resp.citations[0].section.endswith("Verify signatures")


def test_procedural_question_returns_ordered_steps(service):
    resp = service.ask(AskRequest(question="How do I set up the notifications service?"))
    assert resp.answer.splitlines()[0].startswith("1.")
    assert "sender domain" in resp.answer


def test_product_detected_from_question(service):
    resp = service.ask(AskRequest(question="notifications: what is the SMS template limit?"))
    assert all("notifications" in c.source for c in resp.citations)


def test_llm_structured_answer_maps_citations():
    answer = json.dumps({"answer": "Use HMAC-SHA256 over the raw body.", "citations": [1], "confidence": 0.9})
    svc = make_service(FakeListChatModel(responses=[PLAN, answer]))
    resp = svc.ask(AskRequest(question="How do I verify webhook signatures?"))
    assert resp.mode == "llm" and resp.confidence == 0.9
    assert resp.citations[0].title == "Webhooks"


def test_invalid_json_is_retried():
    good = json.dumps({"answer": "Retried answer.", "citations": [1], "confidence": 0.7})
    svc = make_service(FakeListChatModel(responses=[PLAN, "not json at all", good]))
    resp = svc.ask(AskRequest(question="How do I verify webhook signatures?"))
    assert resp.answer == "Retried answer." and resp.mode == "llm"


def test_hallucinated_citation_is_rejected_then_falls_back():
    bad = json.dumps({"answer": "Made up.", "citations": [99], "confidence": 0.99})
    svc = make_service(FakeListChatModel(responses=[PLAN, bad, bad, bad]))
    resp = svc.ask(AskRequest(question="How do I verify webhook signatures?"))
    assert resp.mode == "extractive"
    assert resp.answer != "Made up."
    assert resp.citations


class BrokenLLM:
    def invoke(self, *_args, **_kwargs):
        raise TimeoutError("provider down")


def test_llm_outage_recovers_gracefully():
    svc = make_service(BrokenLLM())
    resp = svc.ask(AskRequest(question="What is the rate limit in live mode?", product="payments"))
    assert resp.mode == "extractive"
    assert "100 requests per second" in resp.answer


def test_unrelated_question_is_not_grounded():
    svc = make_service()
    svc.store._chunks.clear()
    resp = svc.ask(AskRequest(question="What is the capital of France?"))
    assert resp.answer == NOT_FOUND and not resp.grounded and resp.citations == []


def test_stream_emits_steps_then_answer(service):
    events = list(service.ask_stream(AskRequest(question="How long do card refunds take?")))
    nodes = [d["node"] for e, d in events if e == "step"]
    assert nodes[:3] == ["plan", "retrieve", "answer"]
    assert events[-1][0] == "answer" and "5 to 7 business days" in events[-1][1]["answer"]
