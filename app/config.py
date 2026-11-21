import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path | None = None) -> None:
    """Loads KEY=value pairs from .env into the environment. Real env vars win over the file."""
    path = path or Path(os.getenv("DOCPILOT_ENV_FILE", ENV_FILE))
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is in requirements.txt; tolerate a minimal install
        return
    load_dotenv(path, override=False)


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    openai_api_key: str | None
    llm_provider: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None
    llm_timeout: float
    embedding_provider: str
    embedding_model: str | None
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    max_answer_retries: int
    docs_root: str


def load_settings() -> Settings:
    load_env_file()
    return Settings(
        database_url=os.getenv("DATABASE_URL") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_provider=os.getenv("LLM_PROVIDER", "auto"),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL") or None,  # None = the provider's default model
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
        embedding_model=os.getenv("EMBEDDING_MODEL") or None,
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        max_answer_retries=int(os.getenv("MAX_ANSWER_RETRIES", "2")),
        docs_root=os.getenv("DOCS_ROOT", os.getcwd()),
    )
