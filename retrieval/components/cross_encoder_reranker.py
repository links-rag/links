"""Cross-encoder reranking (query + document) on top of retrieval candidates."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Scores (query, passage) pairs with a cross-encoder and returns doc ids sorted by score.

    Loads from a **local directory** (bind-mounted Hugging Face snapshot). No Hub download.

    Note: ``ms-marco-MiniLM-L-6-v2`` overflows to Inf/NaN in float32 under recent
    PyTorch/transformers on CPU (GELU intermediate). Load in float16/bfloat16 instead.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        batch_size: int = 32,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Cross-encoder model directory not found: {path}. "
                "Set paths.cross_encoder in config.yaml to the local model folder."
            )

        if torch_dtype is None:
            # float32 hits Inf→NaN on this checkpoint with torch 2.x; f16/bf16 are stable.
            torch_dtype = torch.float16

        log.info(
            "Loading cross-encoder from local path %s (dtype=%s) ...",
            path,
            torch_dtype,
        )
        self.model_path = str(path)
        self.model: CrossEncoder = CrossEncoder(
            self.model_path,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        self.batch_size = batch_size
        self._smoke_check()

    def _smoke_check(self) -> None:
        """Fail fast if scores are NaN (would silently keep the input ranking)."""
        scores = self.model.predict(
            [("relevance check", "a short relevant passage")],
            show_progress_bar=False,
        )
        val = float(scores[0] if hasattr(scores, "__len__") else scores)
        if not math.isfinite(val):
            raise RuntimeError(
                f"Cross-encoder produced non-finite score ({val}) during smoke check. "
                "Try torch_dtype=torch.float16 (default) or bfloat16; float32 is known "
                "to NaN on ms-marco-MiniLM-L-6-v2 with recent PyTorch."
            )

    def rerank_scored(
        self,
        query: str,
        doc_ids: list[str],
        id_to_text: dict[str, str],
    ) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs, highest score first."""
        if not doc_ids:
            return []
        texts = [id_to_text.get(did, "") or "(empty)" for did in doc_ids]
        pairs = list(zip([query] * len(doc_ids), texts, strict=True))
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        finite_scores: list[float] = []
        for i, s in enumerate(scores):
            val = float(s)
            if not math.isfinite(val):
                raise RuntimeError(
                    f"Cross-encoder score is non-finite ({val}) for doc_id={doc_ids[i]!r}. "
                    "Reranking would silently preserve the RRF order."
                )
            finite_scores.append(val)
        order = sorted(range(len(doc_ids)), key=lambda i: finite_scores[i], reverse=True)
        return [(doc_ids[i], finite_scores[i]) for i in order]

    def rerank(
        self,
        query: str,
        doc_ids: list[str],
        id_to_text: dict[str, str],
    ) -> list[str]:
        """Return ``doc_ids`` reordered by descending relevance score."""
        return [doc_id for doc_id, _score in self.rerank_scored(query, doc_ids, id_to_text)]
