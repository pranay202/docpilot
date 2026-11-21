import os
from pathlib import Path

import pytest

from app.config import load_settings
from app.embeddings import HashingEmbedder
from app.schemas import IngestRequest
from app.service import DocService
from app.store import InMemoryStore

DOCS = Path(__file__).resolve().parent.parent / "sample_docs"

# Tests never read the developer's .env or call a real provider: drop any key from the shell too.
os.environ["DOCPILOT_ENV_FILE"] = os.devnull
for _var in [v for v in os.environ if v.endswith("_API_KEY") or v.startswith("LLM_")]:
    del os.environ[_var]


def make_service(llm=None) -> DocService:
    settings = load_settings()
    svc = DocService(settings, InMemoryStore(), HashingEmbedder(256), llm)
    for product in ("payments", "notifications"):
        svc.ingest(IngestRequest(product=product, source=str(DOCS / product)))
    return svc


@pytest.fixture
def service() -> DocService:
    return make_service()
