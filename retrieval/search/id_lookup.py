"""Fetch one paper when the query is only an arXiv id.

The id is the Chroma row id. This path does not embed the query and does not
call BM25 or the cross-encoder.
"""

from __future__ import annotations

import re

# New ids look like 0704.0001. Old ids look like hep-ph/9901001.
_NEW_ID = re.compile(r"\d{4}\.\d{4,5}")
_OLD_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*/\d{7}")


def exact_arxiv_id(query: str) -> str | None:
    """Return the stored arXiv id when the whole query is that id.

    A bare id, an ``arXiv:`` prefix, a trailing version such as ``v2``, and an
    arxiv.org abs or pdf URL all count. Any other text returns None so a normal
    question still goes through search.

    Args:
        query: The retrieve request query.

    Returns:
        The id stored in Chroma, or None when the query is not only an id.
    """
    text = query.strip()
    text = re.sub(
        r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("/").lower()
    if _NEW_ID.fullmatch(text) or _OLD_ID.fullmatch(text):
        return text
    return None
