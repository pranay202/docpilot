import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.service import DocService, get_service

STATIC = Path(__file__).parent / "static"
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """PRELOAD_DOCS=brand_docs/libreoffice indexes each subfolder as a product at startup."""
    root = os.getenv("PRELOAD_DOCS")
    if root:
        svc = get_service()
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            svc.ingest(IngestRequest(product=folder.name, source=str(folder)))
    yield


app = FastAPI(title="DocPilot", version="0.3.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(svc: DocService = Depends(get_service)) -> dict:
    return {
        "chunks": svc.store.count(),
        "products": svc.store.products(),
        "llm": svc.llm is not None,
        "llm_provider": (svc.llm_info or {}).get("provider"),
        "llm_model": (svc.llm_info or {}).get("model"),
        "embedder": type(svc.embedder).__name__,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, svc: DocService = Depends(get_service)) -> IngestResponse:
    return svc.ingest(req)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, svc: DocService = Depends(get_service)) -> AskResponse:
    return svc.ask(req)


@app.get("/ask/stream")
def ask_stream(
    question: str = Query(min_length=3, max_length=2000),
    product: str | None = None,
    top_k: int = Query(default=5, ge=1, le=20),
    svc: DocService = Depends(get_service),
) -> EventSourceResponse:
    req = AskRequest(question=question, product=product, top_k=top_k)

    def events():
        for event, data in svc.ask_stream(req):
            yield {"event": event, "data": json.dumps(data)}

    return EventSourceResponse(events())
