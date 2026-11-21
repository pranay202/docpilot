import pytest

from app.llm import BY_ID, resolve_embedding_provider, resolve_provider

KEY_ENVS = sorted(
    {e for p in BY_ID.values() for e in p.key_envs} | {"LLM_API_KEY", "LLM_BASE_URL", "EMBEDDING_BASE_URL"}
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env in KEY_ENVS:
        monkeypatch.delenv(env, raising=False)


def test_no_key_means_offline():
    assert resolve_provider("auto") is None
    assert resolve_provider("none") is None


@pytest.mark.parametrize(
    ("env", "provider", "base"),
    [
        ("OPENAI_API_KEY", "openai", None),
        ("GEMINI_API_KEY", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("GOOGLE_API_KEY", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ("OPENROUTER_API_KEY", "openrouter", "https://openrouter.ai/api/v1"),
        ("ANTHROPIC_API_KEY", "anthropic", "https://api.anthropic.com/v1/"),
        ("GROQ_API_KEY", "groq", "https://api.groq.com/openai/v1"),
    ],
)
def test_auto_detects_provider_from_its_key(monkeypatch, env, provider, base):
    monkeypatch.setenv(env, "k-123")
    r = resolve_provider("auto")
    assert r.provider.id == provider and r.api_key == "k-123" and r.base_url == base
    assert r.model == BY_ID[provider].default_model


def test_explicit_provider_with_generic_key_and_model():
    r = resolve_provider("openrouter", api_key="sk-or-x", model="google/gemini-2.5-flash")
    assert r.provider.id == "openrouter" and r.model == "google/gemini-2.5-flash" and r.api_key == "sk-or-x"


def test_explicit_provider_without_key_explains_what_to_set():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        resolve_provider("gemini")


def test_local_providers_need_no_key():
    r = resolve_provider("ollama")
    assert r.base_url == "http://localhost:11434/v1" and r.api_key


def test_custom_needs_base_url_and_model():
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve_provider("custom", model="m")
    r = resolve_provider("auto", base_url="http://box:8000/v1", model="qwen")
    assert r.provider.id == "custom" and r.base_url == "http://box:8000/v1"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        resolve_provider("nope")


@pytest.mark.parametrize(("key", "provider"), [("sk-or-v1-abc", "openrouter"), ("AIzaSyX", "gemini")])
def test_foreign_key_in_openai_var_is_routed_to_its_provider(monkeypatch, key, provider):
    monkeypatch.setenv("OPENAI_API_KEY", key)
    assert resolve_provider("auto").provider.id == provider
    # ...and is never sent to OpenAI's embeddings endpoint.
    assert resolve_embedding_provider("auto", None, key) is None


def test_embeddings_auto_keeps_openai_behaviour(monkeypatch):
    assert resolve_embedding_provider("auto", None, None) is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-1")
    r = resolve_embedding_provider("auto", None, "sk-proj-1")
    assert r.provider.id == "openai" and r.model == "text-embedding-3-small"


def test_gemini_embeddings_are_opt_in(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza1")
    llm = resolve_provider("auto")
    assert resolve_embedding_provider("auto", llm, None) is None
    r = resolve_embedding_provider("gemini", llm, None)
    assert r.model == "gemini-embedding-001" and r.api_key == "AIza1"


def test_build_chat_model_targets_provider_url(monkeypatch):
    from app.llm import build_chat_model

    monkeypatch.setenv("GEMINI_API_KEY", "AIza1")
    chat = build_chat_model(resolve_provider("auto"))
    assert chat.model_name == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in str(chat.openai_api_base)


class _StatusError(Exception):
    def __init__(self, status, body):
        super().__init__(f"Error code: {status}")
        self.status_code, self.body = status, body


GEMINI_BAD_KEY = [
    {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "API_KEY_INVALID"}],
        }
    }
]


def test_gemini_live_bad_key_body_is_reported_as_a_rejected_key():
    """The exact body Gemini's OpenAI endpoint returned for a placeholder key (2026-11-21)."""
    from app.llm import describe_llm_error

    body = [{"error": {"code": 400, "message": "Please pass a valid API key", "status": "INVALID_ARGUMENT"}}]
    f = describe_llm_error(_StatusError(400, body))
    assert f.fatal and "API key was rejected" in f.message


def test_gemini_bad_key_400_is_reported_as_a_rejected_key():
    from app.llm import describe_llm_error

    f = describe_llm_error(_StatusError(400, GEMINI_BAD_KEY))
    assert f.fatal and "API key was rejected" in f.message and "API key not valid" in f.message
    assert "API_KEY_INVALID" in f.message


@pytest.mark.parametrize(
    ("status", "body", "phrase", "fatal"),
    [
        (401, {"error": {"message": "No auth credentials found", "code": 401}}, "API key was rejected", True),
        (402, {"error": {"message": "Insufficient credits"}}, "no credits", True),
        (404, {"error": {"message": "No endpoints found for foo/bar"}}, "model was not found", True),
        (429, {"error": {"message": "Rate limit exceeded"}}, "rate limited", True),
        (503, {"error": {"message": "overloaded"}}, "server error", False),
    ],
)
def test_status_errors_are_classified(status, body, phrase, fatal):
    from app.llm import describe_llm_error

    f = describe_llm_error(_StatusError(status, body))
    assert phrase in f.message and f.fatal is fatal


def test_non_provider_errors_are_left_alone():
    from app.llm import describe_llm_error

    assert describe_llm_error(ValueError("bad json")) is None


class _RejectingLLM:
    calls = 0

    def invoke(self, *_a, **_k):
        type(self).calls += 1
        raise _StatusError(401, {"error": {"message": "invalid key"}})


def test_bad_key_skips_retries_and_explains_the_fallback():
    from app.schemas import AskRequest
    from tests.conftest import make_service

    _RejectingLLM.calls = 0
    svc = make_service(_RejectingLLM())
    resp = svc.ask(AskRequest(question="What is the rate limit in live mode?", product="payments"))
    assert resp.mode == "extractive" and "100 requests per second" in resp.answer
    assert "API key was rejected" in resp.llm_error
    assert _RejectingLLM.calls == 1  # the plan call failed; no answer calls or retries were spent
