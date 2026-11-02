"""Shapes for one search call.

A caller sends a RetrieveRequest. The service answers with a RetrieveResponse.
Default sizes come from config.yaml. No models are loaded here.
"""

from pydantic import BaseModel, Field

from retrieval.config import load_config

__all__ = ["Paper", "RetrieveRequest", "RetrieveResponse", "load_config"]

_RETRIEVAL = load_config()["retrieval"]


class RetrieveRequest(BaseModel):
    """One question to search."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=_RETRIEVAL["top_k"], ge=1)
    categories: list[str] | None = None

    @property
    def candidate_k(self) -> int:
        """How many papers dense search fetches, and how many BM25 fetches."""
        return self.top_k * int(_RETRIEVAL["candidate_multiplier"])

    @property
    def rerank_pool(self) -> int:
        """How many merged papers the cross-encoder rereads."""
        return self.top_k * int(_RETRIEVAL["rerank_multiplier"])


class Paper(BaseModel):
    """One paper returned from search."""

    id: str
    title: str
    text: str
    metadata: dict
    rerank_score: float | None = None
    fused_rank: int | None = None


class RetrieveResponse(BaseModel):
    """Papers for one question, best match first."""

    papers: list[Paper]
