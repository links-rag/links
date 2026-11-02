"""Load Specter2 and Chroma once, then answer many questions in that same process."""

from __future__ import annotations

import sys
from typing import Any

import chromadb

from arxiv_retrieval.config import configured_paths

from .chroma_metadata import build_category_where
from .specter_baseline import Specter2BaselineEmbedder


class EmbeddingSession:
    """Specter2 query embedder with ChromaDB similarity search."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        chroma_path: str | None = None,
        collection_name: str | None = None,
        proximity_adapter_path: str | None = None,
        adhoc_adapter_path: str | None = None,
        device: str | None = None,
    ) -> None:
        if model_path is None or chroma_path is None or collection_name is None:
            paths = configured_paths()
            model_path = model_path or paths["specter_model"]
            chroma_path = chroma_path or paths["chroma"]
            collection_name = collection_name or paths["collection"]

        print(f"Loading Specter2 (adhoc query adapter only) from {model_path}...", file=sys.stderr)
        self.embedder = Specter2BaselineEmbedder(
            model_path,
            adapters="query",
            proximity_adapter_path=proximity_adapter_path,
            adhoc_adapter_path=adhoc_adapter_path,
            device=device,
        )

        print(f"Opening ChromaDB at {chroma_path}...", file=sys.stderr)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"Chroma collection '{collection_name}' ready ({self.collection.count()} records).", file=sys.stderr)

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query with the adhoc adapter."""
        return self.embedder.embed_query([text])[0]

    def search(
        self,
        query: str,
        top_k: int = 10,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embed_query(query)
        where = build_category_where(categories)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        papers: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            distance = distances[idx] if idx < len(distances) else None
            score = 1.0 - distance if distance is not None else None

            papers.append(
                {
                    "id": doc_id,
                    "title": metadata.get("title", ""),
                    "abstract": documents[idx] if idx < len(documents) else "",
                    "metadata": metadata,
                    "score": score,
                }
            )

        return papers
