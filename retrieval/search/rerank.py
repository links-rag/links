"""Cross-encoder: reread the front of the merged list and keep the best papers.

The merge chooses which papers reach this step. This step only reorders them.
"""

from __future__ import annotations

from pathlib import Path

from ..components.cross_encoder_reranker import CrossEncoderReranker
from ..paths import REPO_ROOT
from ..schemas import RetrieveRequest, load_config


class CrossEncoderSearch:
    """Loads the cross-encoder once, then reranks many merged lists."""

    def __init__(self, model_path: Path) -> None:
        self.reranker = CrossEncoderReranker(model_path)

    @classmethod
    def from_config(cls) -> "CrossEncoderSearch":
        """Load the local model folder named in config.yaml."""
        paths = load_config()["paths"]
        return cls(REPO_ROOT / paths["cross_encoder"])

    def rerank(self, request: RetrieveRequest, merged: list[dict]) -> list[dict]:
        """Reread rerank_pool papers and return the top_k best.

        For top_k 100 that is 200 papers in and 100 papers out.
        fused_rank stays the place from the merge. rerank_score is the new score.
        """
        pool = merged[: request.rerank_pool]
        by_id = {row["id"]: row for row in pool}
        scored = self.reranker.rerank_scored(
            request.query,
            [row["id"] for row in pool],
            {row["id"]: row.get("text", "") for row in pool},
        )

        kept: list[dict] = []
        for doc_id, score in scored[: request.top_k]:
            row = dict(by_id[doc_id])
            row["rerank_score"] = score
            kept.append(row)
        return kept
