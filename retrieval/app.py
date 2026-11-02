"""HTTP door to the retrieval pipeline.

GET /health is 503 until Specter2, Chroma, BM25, and the cross-encoder are loaded.
POST /v1/retrieve returns the papers the judge should read.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .schemas import RetrieveRequest, RetrieveResponse
from .search.pipeline import RetrievalPipeline

_pipeline: RetrievalPipeline | None = None
_load_error: str | None = None


def _load_pipeline() -> None:
    """Load the models off the request thread so /health can answer meanwhile."""
    global _pipeline, _load_error
    try:
        _pipeline = RetrievalPipeline.from_config()
    except Exception as exc:
        _load_error = str(exc)
        _pipeline = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start model loading when the server starts, and leave it running."""
    worker = threading.Thread(target=_load_pipeline, daemon=True)
    worker.start()
    yield


app = FastAPI(title="Retrieval service", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    """Ready only after the search models have loaded."""
    if _pipeline is None:
        detail = _load_error or "loading"
        raise HTTPException(status_code=503, detail=detail)
    return {"status": "ready"}


@app.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Run dense search, BM25, the merge, and the cross-encoder.

    A query that is only an arXiv id returns that paper without searching.
    """
    if _pipeline is None:
        detail = _load_error or "loading"
        raise HTTPException(status_code=503, detail=detail)
    return _pipeline.retrieve(request)
