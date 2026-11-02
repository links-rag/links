"""Lexical BM25 index plus reciprocal rank fusion with dense id lists."""

from __future__ import annotations

import logging
from typing import Iterable

import bm25s

log = logging.getLogger(__name__)


class BM25CorpusIndex:
    """In-memory BM25 index over document strings read from Chroma."""

    def __init__(self, id_to_text: dict[str, str]) -> None:
        """Build an index from paper id to document text.

        Args:
            id_to_text: Paper id to the text stored in Chroma.
        """
        self.ids = list(id_to_text.keys())
        self.texts = [id_to_text[i] for i in self.ids]
        log.info("BM25 indexing %d documents ...", len(self.texts))
        tokens = bm25s.tokenize(self.texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens)

    def topk_ids(self, query: str, k: int) -> list[str]:
        """Return the k paper ids with the strongest word match.

        Args:
            query: Question text.
            k: How many ids to return.

        Returns:
            Paper ids, best word match first.
        """
        k = max(1, min(k, len(self.ids)))
        qt = bm25s.tokenize([query])
        idx, _scores = self.retriever.retrieve(qt, k=k)
        row = idx[0]
        return [self.ids[int(i)] for i in row]


def rrf_fuse_rankings(
    ranked_lists: Iterable[list[str]],
    *,
    rrf_k: float = 60.0,
    max_depth: int = 100,
) -> list[str]:
    """Combine ordered paper-id lists into one list.

    Args:
        ranked_lists: Ordered id lists, best match first in each list.
        rrf_k: Constant added to each rank before taking the reciprocal.
        max_depth: How many ids to read from the front of each list.

    Returns:
        Paper ids, highest fused score first.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(list(lst)[:max_depth], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
