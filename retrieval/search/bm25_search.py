"""BM25 search: match the question against paper words.

The word list is built in memory from Chroma when this object is created.
Questions after that only look words up. Nothing is written back to Chroma.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import chromadb

from ..components.chroma_metadata import category_metadata_key
from ..components.hybrid_rrf import BM25CorpusIndex
from ..paths import REPO_ROOT
from ..schemas import RetrieveRequest, load_config

# Chroma's one-shot get() binds one SQL variable per paper and fails past
# SQLite's limit. This collection has about 250,000 papers.
_GET_BATCH = 500


class Bm25Search:
    """One Chroma collection, plus a word list kept in memory."""

    def __init__(self, chroma: Path, collection: str) -> None:
        self._chroma_path = Path(chroma)
        client = chromadb.PersistentClient(path=str(self._chroma_path))
        self.collection = client.get_collection(name=collection)
        self._indexes: dict[tuple[str, ...], BM25CorpusIndex] = {}
        self._rows: dict[tuple[str, ...], dict[str, dict]] = {}
        self._index_for(None)

    @classmethod
    def from_config(cls) -> "Bm25Search":
        """Open the Chroma path in config.yaml and build the word list."""
        paths = load_config()["paths"]
        return cls(
            chroma=REPO_ROOT / paths["chroma"],
            collection=paths["collection"],
        )

    def search(self, request: RetrieveRequest) -> list[dict]:
        """Return candidate_k papers, best word match first."""
        key = self._index_for(request.categories)
        index = self._indexes[key]
        rows = self._rows[key]
        ranked: list[dict] = []
        for place, doc_id in enumerate(index.topk_ids(request.query, request.candidate_k), start=1):
            row = rows[doc_id]
            ranked.append(
                {
                    "id": doc_id,
                    "title": row["title"],
                    "text": row["text"],
                    "metadata": row["metadata"],
                    "bm25_rank": place,
                }
            )
        return ranked

    def get_paper(self, arxiv_id: str) -> dict | None:
        """Load one paper by the arXiv id stored as its Chroma id.

        Args:
            arxiv_id: Id such as ``0704.0001`` or ``hep-ph/9901001``.

        Returns:
            The paper dict, or None when that id is not in the collection.
        """
        result = self.collection.get(ids=[arxiv_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids or not ids[0]:
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        document = documents[0] if documents else ""
        metadata = metadatas[0] if metadatas and metadatas[0] else {}
        if not document:
            return None
        return {
            "id": str(ids[0]),
            "title": metadata.get("title", ""),
            "text": str(document),
            "metadata": metadata,
        }

    def _index_for(self, categories: list[str] | None) -> tuple[str, ...]:
        """Build the word list once for this category filter, then reuse it."""
        key = tuple(sorted(categories or []))
        if key in self._indexes:
            return key
        rows = self._load_rows(categories)
        self._rows[key] = rows
        self._indexes[key] = BM25CorpusIndex(
            {doc_id: row["text"] for doc_id, row in rows.items()}
        )
        return key

    def _load_rows(self, categories: list[str] | None) -> dict[str, dict]:
        """Read id, title, text, and metadata from Chroma in small batches."""
        paper_ids = _paper_ids(self._chroma_path, categories)
        rows: dict[str, dict] = {}
        for start in range(0, len(paper_ids), _GET_BATCH):
            chunk = paper_ids[start : start + _GET_BATCH]
            result = self.collection.get(ids=chunk, include=["documents", "metadatas"])
            ids = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for doc_id, document, metadata in zip(ids, documents, metadatas, strict=False):
                if not doc_id or not document:
                    continue
                metadata = metadata or {}
                rows[str(doc_id)] = {
                    "title": metadata.get("title", ""),
                    "text": str(document),
                    "metadata": metadata,
                }
        return rows


def _paper_ids(chroma_path: Path, categories: list[str] | None) -> list[str]:
    """List arXiv ids from the Chroma sqlite file, optionally by category.

    A direct read avoids asking Chroma to bind every paper id in one statement.
    """
    keys = [
        category_metadata_key(category.strip())
        for category in (categories or [])
        if category and category.strip()
    ]
    database = chroma_path / "chroma.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        if not keys:
            cursor = connection.execute("SELECT embedding_id FROM embeddings")
        else:
            placeholders = ",".join("?" for _ in keys)
            cursor = connection.execute(
                f"""
                SELECT DISTINCT e.embedding_id
                FROM embeddings e
                JOIN embedding_metadata m ON m.id = e.id
                WHERE m.bool_value = 1 AND m.key IN ({placeholders})
                """,
                keys,
            )
        return [str(row[0]) for row in cursor if row[0]]
    finally:
        connection.close()
