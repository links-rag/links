"""Merge the Specter2 list and the BM25 list into one order.

This step does not read abstracts again and it does not call the cross-encoder.
A paper near the top of both lists ends up above a paper that only one list liked.
"""

from __future__ import annotations

from ..components.hybrid_rrf import rrf_fuse_rankings
from ..schemas import load_config


def merge_rankings(
    dense: list[dict],
    bm25: list[dict],
    *,
    rrf_k: float | None = None,
) -> list[dict]:
    """Return one list, best merged paper first.

    Each paper keeps dense_rank when Specter2 found it, and bm25_rank when
    BM25 found it. fused_rank is its place after the merge, starting at 1.
    """
    if rrf_k is None:
        rrf_k = float(load_config()["retrieval"]["rrf_k"])

    by_id: dict[str, dict] = {}
    for hit in dense:
        by_id[hit["id"]] = dict(hit)
    for hit in bm25:
        row = by_id.get(hit["id"])
        if row is None:
            by_id[hit["id"]] = dict(hit)
        else:
            row["bm25_rank"] = hit["bm25_rank"]

    order = rrf_fuse_rankings(
        [[hit["id"] for hit in dense], [hit["id"] for hit in bm25]],
        rrf_k=rrf_k,
        max_depth=max(len(dense), len(bm25), 1),
    )

    merged: list[dict] = []
    for place, doc_id in enumerate(order, start=1):
        row = dict(by_id[doc_id])
        row["fused_rank"] = place
        merged.append(row)
    return merged
