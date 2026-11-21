"""LLM and embedding provider registry.

Every provider below exposes an OpenAI-compatible API, so a single client
(`langchain_openai.ChatOpenAI` / `OpenAIEmbeddings`) pointed at the right base URL
covers all of them. No provider-specific SDK is needed.

Selection (see `resolve_provider`):
- `LLM_PROVIDER=<id>` picks one explicitly.
- `LLM_PROVIDER=auto` (the default) picks the first provider whose API key env var is set,
  in the order of `PROVIDERS`.
- `LLM_PROVIDER=custom` + `LLM_BASE_URL` targets any other OpenAI-compatible server.

The key is read from `LLM_API_KEY`, else the provider's own env vars (e.g. `GEMINI_API_KEY`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger("docpilot.llm")


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    base_url: str | None  # None = the client's default (api.openai.com)
    key_envs: tuple[str, ...]
    default_model: str
    default_embedding_model: str | None = None  # None = provider has no embeddings endpoint we support
    needs_key: bool = True  # local servers accept any placeholder key


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "OpenAI", None, ("OPENAI_API_KEY",), "gpt-4o-mini", "text-embedding-3-small"),
    Provider(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-2.5-flash",
        "gemini-embedding-001",
    ),
    Provider(
        "anthropic", "Anthropic Claude", "https://api.anthropic.com/v1/", ("ANTHROPIC_API_KEY",), "claude-haiku-5-5"
    ),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",), "openai/gpt-4o-mini"),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",), "llama-3.3-70b-versatile"),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat"),
    Provider(
        "together",
        "Together AI",
        "https://api.together.xyz/v1",
        ("TOGETHER_API_KEY",),
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ),
    Provider(
        "mistral", "Mistral", "https://api.mistral.ai/v1", ("MISTRAL_API_KEY",), "mistral-small-latest", "mistral-embed"
    ),
    Provider(
        "ollama", "Ollama (local)", "http://localhost:11434/v1", (), "llama3.1", "nomic-embed-text", needs_key=False
    ),
    Provider("lmstudio", "LM Studio (local)", "http://localhost:1234/v1", (), "local-model", needs_key=False),
    Provider("custom", "Custom OpenAI-compatible", None, (), "", needs_key=False),
)
BY_ID = {p.id: p for p in PROVIDERS}
# Embedding endpoints that honour the `dimensions` parameter (so EMBEDDING_DIM can be anything).
DIMENSION_AWARE = frozenset({"openai", "gemini"})
# Keys people commonly paste into OPENAI_API_KEY that belong to another provider.
FOREIGN_KEY_PREFIXES = {"sk-or-": "openrouter", "AIza": "gemini", "sk-ant-": "anthropic", "gsk_": "groq"}


def misplaced_openai_key() -> str | None:
    """The provider id an OPENAI_API_KEY value really belongs to, if it is recognisably not OpenAI's."""
    value = os.getenv("OPENAI_API_KEY") or ""
    return next((pid for prefix, pid in FOREIGN_KEY_PREFIXES.items() if value.startswith(prefix)), None)


# Local/custom servers are never auto-detected: they have no key to detect them by.
AUTO_ORDER = tuple(p for p in PROVIDERS if p.key_envs)


@dataclass(frozen=True)
class ResolvedLLM:
    provider: Provider
    model: str
    api_key: str
    base_url: str | None


def _key_for(provider: Provider, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for env in provider.key_envs:
        if value := os.getenv(env):
            return value
    return None


def resolve_provider(
    provider_id: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ResolvedLLM | None:
    """Returns the provider to use, or None for offline (extractive) mode."""
    provider_id = (provider_id or "auto").strip().lower()
    if provider_id in ("none", "off", "offline", "extractive"):
        return None
    if provider_id == "auto":
        if base_url:  # a base URL with no provider named means "custom server"
            provider_id = "custom"
        elif (real := misplaced_openai_key()) and not api_key:
            log.warning("OPENAI_API_KEY holds a key for %s, so %s is used", BY_ID[real].label, BY_ID[real].label)
            provider_id = real
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            for candidate in AUTO_ORDER:
                if _key_for(candidate, None):
                    provider_id = candidate.id
                    break
            else:
                return None
    provider = BY_ID.get(provider_id)
    if provider is None:
        raise ValueError(f"Unknown LLM_PROVIDER {provider_id!r}. Choose one of: auto, {', '.join(BY_ID)}, none")
    key = _key_for(provider, api_key)
    if not key:
        if provider.needs_key:
            envs = " or ".join(("LLM_API_KEY", *provider.key_envs))
            raise ValueError(f"LLM_PROVIDER={provider.id} needs an API key: set {envs}")
        key = "not-needed"
    url = base_url or provider.base_url
    if provider.id == "custom" and not url:
        raise ValueError("LLM_PROVIDER=custom needs LLM_BASE_URL (an OpenAI-compatible /v1 URL)")
    chosen_model = model or provider.default_model
    if not chosen_model:
        raise ValueError(f"LLM_PROVIDER={provider.id} needs LLM_MODEL")
    return ResolvedLLM(provider, chosen_model, key, url)


def build_chat_model(resolved: ResolvedLLM, timeout: float = 30):
    from langchain_openai import ChatOpenAI

    kwargs: dict = {}
    if resolved.provider.id == "openrouter":
        # Optional attribution headers OpenRouter documents; harmless elsewhere.
        kwargs["default_headers"] = {"X-Title": "DocPilot"}
    log.info("LLM: %s, model %s", resolved.provider.label, resolved.model)
    return ChatOpenAI(
        model=resolved.model,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        temperature=0,
        timeout=timeout,
        max_retries=1,
        **kwargs,
    )


def resolve_embedding_provider(
    embedding_provider: str | None,
    llm: ResolvedLLM | None,
    openai_key: str | None,
    model: str | None = None,
) -> ResolvedLLM | None:
    """Which provider serves embeddings, or None for the offline hashing embedder.

    `auto` keeps the historical behaviour: OpenAI embeddings when OPENAI_API_KEY is set,
    otherwise hashing. Other providers' embeddings are opt-in (EMBEDDING_PROVIDER=gemini, ...),
    because switching embedders changes every stored vector and ingest makes many API calls.
    """
    choice = (embedding_provider or "auto").strip().lower()
    if choice in ("hashing", "none", "offline"):
        return None
    if choice == "auto":
        if not openai_key or misplaced_openai_key():
            return None
        choice = "openai"
    if llm is not None and llm.provider.id == choice:
        provider, key, url = llm.provider, llm.api_key, llm.base_url
    else:
        provider = BY_ID.get(choice)
        if provider is None:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER {choice!r}")
        key = _key_for(provider, None) or ("not-needed" if not provider.needs_key else None)
        url = os.getenv("EMBEDDING_BASE_URL") or provider.base_url
        if not key:
            raise ValueError(f"EMBEDDING_PROVIDER={choice} needs {' or '.join(provider.key_envs)}")
    chosen = model or provider.default_embedding_model
    if not chosen:
        raise ValueError(f"{provider.label} has no default embedding model: set EMBEDDING_MODEL")
    return ResolvedLLM(provider, chosen, key, url)


@dataclass(frozen=True)
class LLMFailure:
    """A provider error turned into something a person can act on."""

    message: str
    fatal: bool  # retrying the same request cannot succeed (bad key, unknown model, no credits)


def _error_text(exc: Exception) -> str:
    """The provider's own error message, dug out of the response body when there is one."""
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:  # Gemini wraps its error in a one-element list
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            reasons = [d.get("reason") for d in err.get("details") or [] if isinstance(d, dict) and d.get("reason")]
            msg = str(err.get("message") or "")
            return f"{msg} ({', '.join(reasons)})" if reasons else msg
        if isinstance(err, str):
            return err
    return str(getattr(exc, "message", None) or exc)


def describe_llm_error(exc: Exception) -> LLMFailure | None:
    """Classifies a provider/network error, or returns None when `exc` is not one."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status is None:
        if "Timeout" in name:
            return LLMFailure("the provider timed out (raise LLM_TIMEOUT if this keeps happening)", fatal=False)
        if "Connection" in name:
            return LLMFailure(
                "could not reach the provider: check your network, or LLM_BASE_URL if you set one", fatal=True
            )
        return None
    detail = _error_text(exc).strip()[:300]
    lowered = detail.lower()
    bad_key = any(m in lowered for m in ("api_key_invalid", "api key not valid", "valid api key", "invalid api key"))
    if status in (401, 403) or bad_key:
        return LLMFailure(
            f"the API key was rejected (HTTP {status}: {detail}). Put a valid key in .env and restart the server",
            fatal=True,
        )
    if status == 402:
        return LLMFailure(f"the account has no credits left (HTTP 402: {detail})", fatal=True)
    if status == 404 or "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return LLMFailure(f"the model was not found (HTTP {status}: {detail}). Check LLM_MODEL", fatal=True)
    if status == 429:
        return LLMFailure(f"rate limited or out of quota (HTTP 429: {detail})", fatal=True)
    if status >= 500:
        return LLMFailure(f"the provider had a server error (HTTP {status}: {detail})", fatal=False)
    return LLMFailure(f"the provider refused the request (HTTP {status}: {detail})", fatal=status == 400)
