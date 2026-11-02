"""LlamaIndex wrapper around the four search steps.

Dense search, BM25, the merge, and the cross-encoder stay in their own files.
This module only calls them, in that order, and returns nodes.
"""

from __future__ import annotations

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import PrivateAttr

from ..schemas import Paper, RetrieveRequest, RetrieveResponse
from ..search.bm25_search import Bm25Search
from ..search.dense_search import DenseSearch
from ..search.id_lookup import exact_arxiv_id
from ..search.merge import merge_rankings
from ..search.rerank import CrossEncoderSearch

_RANK_KEYS = ("dense_rank", "bm25_rank", "fused_rank", "rerank_score", "title")


def _metadata(hit: dict) -> dict:
    """Keep rank fields, and copy Chroma metadata that a node can store."""
    meta: dict = {"title": hit.get("title") or ""}
    for key in ("dense_rank", "bm25_rank", "fused_rank"):
        if hit.get(key) is not None:
            meta[key] = int(hit[key])
    if hit.get("rerank_score") is not None:
        meta["rerank_score"] = float(hit["rerank_score"])
    for key, value in (hit.get("metadata") or {}).items():
        if key in meta:
            continue
        if isinstance(value, (str, int, float)) or value is None:
            meta[str(key)] = value
        else:
            meta[str(key)] = str(value)
    return meta


def paper_to_node(hit: dict) -> NodeWithScore:
    """One search hit becomes one LlamaIndex node."""
    score = hit.get("rerank_score")
    node = TextNode(
        id_=str(hit["id"]),
        text=hit.get("text") or "",
        metadata=_metadata(hit),
    )
    return NodeWithScore(node=node, score=None if score is None else float(score))


def node_to_hit(item: NodeWithScore) -> dict:
    """Turn a node back into the dict the cross-encoder already expects."""
    meta = dict(item.node.metadata or {})
    chroma_meta = {key: value for key, value in meta.items() if key not in _RANK_KEYS}
    return {
        "id": item.node.node_id,
        "title": meta.get("title", ""),
        "text": item.node.get_content(),
        "metadata": chroma_meta,
        "dense_rank": meta.get("dense_rank"),
        "bm25_rank": meta.get("bm25_rank"),
        "fused_rank": meta.get("fused_rank"),
        "rerank_score": meta.get("rerank_score"),
    }


class HybridRetriever(BaseRetriever):
    """Specter2 list plus BM25 list, merged with reciprocal rank fusion."""

    def __init__(self, dense: DenseSearch, bm25: Bm25Search) -> None:
        super().__init__()
        self._dense = dense
        self._bm25 = bm25

    def retrieve_request(self, request: RetrieveRequest) -> list[NodeWithScore]:
        """Run both searches for this request, including its category filter."""
        dense_hits = self._dense.search(request)
        bm25_hits = self._bm25.search(request)
        merged = merge_rankings(dense_hits, bm25_hits)
        return [paper_to_node(hit) for hit in merged]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self.retrieve_request(RetrieveRequest(query=query_bundle.query_str))


class CrossEncoderPostprocessor(BaseNodePostprocessor):
    """Reread the front of the merged list and keep top_k papers."""

    _reranker: CrossEncoderSearch = PrivateAttr()

    def __init__(self, reranker: CrossEncoderSearch, **kwargs) -> None:
        super().__init__(**kwargs)
        self._reranker = reranker

    def rerank_request(
        self,
        request: RetrieveRequest,
        nodes: list[NodeWithScore],
    ) -> list[NodeWithScore]:
        """Use this request's top_k, so 100 papers stay 100 papers."""
        kept = self._reranker.rerank(request, [node_to_hit(item) for item in nodes])
        return [paper_to_node(hit) for hit in kept]

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        query = query_bundle.query_str if query_bundle is not None else ""
        return self.rerank_request(RetrieveRequest(query=query), nodes)


class RetrievalPipeline:
    """One object FastAPI can call. It does not write the Gemma answer."""

    def __init__(
        self,
        dense: DenseSearch,
        bm25: Bm25Search,
        reranker: CrossEncoderSearch,
    ) -> None:
        self.bm25 = bm25
        self.retriever = HybridRetriever(dense, bm25)
        self.postprocessor = CrossEncoderPostprocessor(reranker)

    @classmethod
    def from_config(cls) -> "RetrievalPipeline":
        """Load Specter2, Chroma, the BM25 word list, and the cross-encoder."""
        return cls(
            dense=DenseSearch.from_config(),
            bm25=Bm25Search.from_config(),
            reranker=CrossEncoderSearch.from_config(),
        )

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Return the papers the judge should read, best match first.

        A query that is only an arXiv id loads that paper from Chroma and
        skips dense search, BM25, the merge, and the cross-encoder.
        """
        arxiv_id = exact_arxiv_id(request.query)
        if arxiv_id is not None:
            return RetrieveResponse(papers=self._papers_for_id(arxiv_id))
        nodes = self.retriever.retrieve_request(request)
        nodes = self.postprocessor.rerank_request(request, nodes)
        papers = []
        for item in nodes:
            hit = node_to_hit(item)
            papers.append(
                Paper(
                    id=hit["id"],
                    title=hit["title"],
                    text=hit["text"],
                    metadata=hit["metadata"],
                    rerank_score=hit.get("rerank_score"),
                    fused_rank=hit.get("fused_rank"),
                )
            )
        return RetrieveResponse(papers=papers)

    def _papers_for_id(self, arxiv_id: str) -> list[Paper]:
        """One paper for a direct id lookup, or an empty list when it is missing."""
        hit = self.bm25.get_paper(arxiv_id)
        if hit is None:
            return []
        return [
            Paper(
                id=hit["id"],
                title=hit["title"],
                text=hit["text"],
                metadata=hit["metadata"],
            )
        ]
