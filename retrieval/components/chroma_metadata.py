"""Shared ChromaDB metadata helpers for arXiv category filtering."""

from __future__ import annotations

from typing import Any


def category_metadata_key(category: str) -> str:
    """Map an arXiv category to a Chroma-safe metadata key."""
    return "cat_" + category.replace(".", "_").replace("-", "_")


def build_category_where(categories: list[str] | None) -> dict[str, Any] | None:
    """Build a Chroma ``where`` filter for one or more arXiv categories."""
    if not categories:
        return None

    requested = [category.strip() for category in categories if category.strip()]
    if not requested:
        return None

    clauses = [{category_metadata_key(category): True} for category in requested]
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}

