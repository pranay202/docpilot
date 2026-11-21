import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    openai_api_key: str | None
    llm_model: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    max_answer_retries: int
    docs_root: str


def load_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "384")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        max_answer_retries=int(os.getenv("MAX_ANSWER_RETRIES", "2")),
        docs_root=os.getenv("DOCS_ROOT", os.getcwd()),
    )
