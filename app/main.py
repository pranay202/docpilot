from fastapi import FastAPI, HTTPException

from app.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse

app = FastAPI(title="DocPilot", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest(req: IngestRequest) -> IngestResponse:
    # Milestone: load, chunk and embed the source into pgvector.
    return IngestResponse(product=req.product, source=req.source, status="queued")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # Milestone: route through the LangGraph agent (plan -> retrieve -> answer -> verify).
    raise HTTPException(status_code=501, detail="Agent not implemented yet")
