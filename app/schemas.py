from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    product: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    source: str = Field(min_length=1, description="Markdown folder path or sitemap URL")


class IngestResponse(BaseModel):
    product: str
    source: str
    status: Literal["done", "failed"]
    documents: int = 0
    chunks: int = 0
    detail: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    product: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    title: str
    section: str
    source: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    mode: Literal["llm", "extractive"] = "extractive"
    grounded: bool = True
    llm_error: str | None = None  # why an LLM was configured but not used for this answer


class LLMDraft(BaseModel):
    """The JSON shape the model must return. Citations are context block numbers."""

    answer: str = Field(min_length=1)
    citations: list[int] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
