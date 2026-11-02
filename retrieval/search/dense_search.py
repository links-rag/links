"""Dense search: Specter2 embeds the question, Chroma returns the nearest papers.

This is one of the two lists that will later be merged with BM25.
It does not rerank and it does not call Gemma.
"""

from __future__ import annotations

from pathlib import Path

from ..components.embed_session import EmbeddingSession
from ..paths import REPO_ROOT
from ..schemas import RetrieveRequest, load_config


class DenseSearch:
    """Loads Specter2 and Chroma once, then answers many questions."""

    def __init__(self, specter_model: Path, chroma: Path, collection: str) -> None:
        self.session = EmbeddingSession(
            model_path=str(specter_model),
            chroma_path=str(chroma),
            collection_name=collection,
        )

    @classmethod
    def from_config(cls) -> "DenseSearch":
        """Build from the paths in config.yaml, relative to the repo root."""
        paths = load_config()["paths"]
        return cls(
            specter_model=REPO_ROOT / paths["specter_model"],
            chroma=REPO_ROOT / paths["chroma"],
            collection=paths["collection"],
        )

    def search(self, request: RetrieveRequest) -> list[dict]:
        """Return candidate_k papers, nearest first.

        candidate_k is 3 times top_k. For top_k 100 that is 300 papers.
        """
        hits = self.session.search(
            request.query,
            top_k=request.candidate_k,
            categories=request.categories,
        )
        ranked: list[dict] = []
        for place, hit in enumerate(hits, start=1):
            ranked.append(
                {
                    "id": hit["id"],
                    "title": hit["title"],
                    "text": hit["abstract"],
                    "metadata": hit["metadata"],
                    "dense_rank": place,
                }
            )
        return ranked
