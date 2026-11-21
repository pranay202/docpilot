from pathlib import Path

import pytest

from app.config import load_settings
from app.embeddings import HashingEmbedder
from app.schemas import IngestRequest
from app.service import DocService
from app.store import InMemoryStore

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"


def make_service(llm=None) -> DocService:
    settings = load_settings()
    svc = DocService(settings, InMemoryStore(), HashingEmbedder(256), llm)
    for product in ("payments", "notifications"):
        svc.ingest(IngestRequest(product=product, source=str(DOCS / product)))
    return svc


@pytest.fixture
def service() -> DocService:
    return make_service()
