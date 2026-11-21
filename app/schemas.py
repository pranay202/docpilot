from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class IngestRequest(BaseModel):
    product: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, description="Markdown folder path or sitemap URL")


class IngestResponse(BaseModel):
    product: str
    source: str
    status: Literal["queued", "done", "failed"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    product: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    title: str
    url: HttpUrl | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
