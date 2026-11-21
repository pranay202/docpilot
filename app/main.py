import json
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.service import DocService, get_service

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="DocPilot", version="0.2.0")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(svc: DocService = Depends(get_service)) -> dict:
    return {"chunks": svc.store.count(), "products": svc.store.products(), "llm": svc.llm is not None}


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
