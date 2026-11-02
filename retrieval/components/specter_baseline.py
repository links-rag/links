"""SPECTER2 baseline embeddings for Chroma ingestion and ad-hoc query retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

DEFAULT_MAX_EMBED_TOKENS = 512
DEFAULT_CHUNK_STRIDE = 256

# `load_as` names — must match load_adapter(..., load_as=...).
_ADAPTER_PROXIMITY = "proximity"  # allenai/specter2 — title+abstract / corpus indexing
_ADAPTER_ADHOC = "adhoc"  # allenai/specter2_adhoc_query — user queries at retrieval

_AdapterKey = Literal["proximity", "adhoc"]
AdapterMode = Literal["corpus", "query", "both"]

_BASE_DIR_NAMES = ("specter2_base", "base")
_PROXIMITY_DIR_NAMES = ("specter2", "proximity", "adapter_proximity")
_ADHOC_DIR_NAMES = ("specter2_adhoc_query", "adhoc", "adapter_adhoc")


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _first_existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def resolve_specter2_paths(
    model_path: str | Path,
    *,
    proximity_adapter_path: str | Path | None = None,
    adhoc_adapter_path: str | Path | None = None,
    require_proximity: bool = True,
    require_adhoc: bool = True,
) -> tuple[Path, Path | None, Path | None]:
    """
    Resolve local Specter2 base + adapter directories.

    ``model_path`` may be:
      - the base model directory itself (contains ``config.json``)
      - a parent directory containing ``specter2_base/``, ``specter2/``,
        and ``specter2_adhoc_query/`` subdirectories
    """
    root = Path(model_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"model path does not exist: {root}")

    if _is_model_dir(root):
        base = root
        search_roots = [root.parent, root]
    else:
        base = _first_existing_dir(root, _BASE_DIR_NAMES)
        if base is None and _is_model_dir(root):
            base = root
        if base is None:
            raise FileNotFoundError(
                f"Could not find specter2 base model under {root}. "
                "Expected config.json in the path or a specter2_base/ subdirectory."
            )
        search_roots = [root, root.parent]

    proximity: Path | None
    if not require_proximity:
        proximity = None
    elif proximity_adapter_path is not None:
        proximity = Path(proximity_adapter_path).resolve()
    else:
        proximity = None
        for search_root in search_roots:
            proximity = _first_existing_dir(search_root, _PROXIMITY_DIR_NAMES)
            if proximity is not None:
                break

    adhoc: Path | None
    if not require_adhoc:
        adhoc = None
    elif adhoc_adapter_path is not None:
        adhoc = Path(adhoc_adapter_path).resolve()
    else:
        adhoc = None
        for search_root in search_roots:
            adhoc = _first_existing_dir(search_root, _ADHOC_DIR_NAMES)
            if adhoc is not None:
                break

    if require_proximity and (proximity is None or not proximity.is_dir()):
        raise FileNotFoundError(
            "Proximity adapter not found. Pass --proximity-adapter-path or place "
            "specter2/ next to the base model."
        )
    if require_adhoc and (adhoc is None or not adhoc.is_dir()):
        raise FileNotFoundError(
            "Adhoc adapter not found. Pass --adhoc-adapter-path or place "
            "specter2_adhoc_query/ next to the base model."
        )

    return base, proximity, adhoc


def count_embedding_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Token count for ``text`` using the same settings as embedding (with special tokens)."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def split_text_for_embedding(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int = DEFAULT_MAX_EMBED_TOKENS,
    stride: int = DEFAULT_CHUNK_STRIDE,
) -> tuple[int, list[str]]:
    """
    Return total token count and one or more text chunks for embedding.

    Short texts are returned unchanged. Longer texts are split with HuggingFace
    ``return_overflowing_tokens`` (``max_length`` tokens per chunk, ``stride`` overlap).
    """
    token_count = count_embedding_tokens(tokenizer, text)
    if token_count <= max_length:
        return token_count, [text]

    overflow = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=True,
        stride=stride,
    )
    chunk_ids = overflow["input_ids"]
    if chunk_ids and not isinstance(chunk_ids[0], list):
        chunk_ids = [chunk_ids]
    chunk_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in chunk_ids]
    return token_count, chunk_texts


class Specter2BaselineEmbedder:
    """
    SPECTER2 base with separate corpus and query adapter paths.

    - ``adapters="corpus"``  — proximity only (title+abstract indexing / Chroma ingest)
    - ``adapters="query"``   — adhoc only (retrieval queries)
    - ``adapters="both"``    — load both adapters
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        adapters: AdapterMode = "both",
        proximity_adapter_path: str | Path | None = None,
        adhoc_adapter_path: str | Path | None = None,
        device: str | None = None,
        adapter_checkpoint: str | Path | None = None,
        max_embed_tokens: int = DEFAULT_MAX_EMBED_TOKENS,
        chunk_stride: int = DEFAULT_CHUNK_STRIDE,
    ) -> None:
        load_proximity = adapters in {"corpus", "both"}
        load_adhoc = adapters in {"query", "both"}
        self.max_embed_tokens = max_embed_tokens
        self.chunk_stride = chunk_stride

        base, proximity, adhoc = resolve_specter2_paths(
            model_path,
            proximity_adapter_path=proximity_adapter_path,
            adhoc_adapter_path=adhoc_adapter_path,
            require_proximity=load_proximity,
            require_adhoc=load_adhoc,
        )
        self.adapters = adapters
        self.model_name = str(base)
        self.proximity_adapter_path = str(proximity) if proximity is not None else None
        self.adhoc_adapter_path = str(adhoc) if adhoc is not None else None
        self.device = _pick_device(device)

        log.info(
            "Loading tokenizer and model from %s on %s (adapters=%s) ...",
            base,
            self.device,
            adapters,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.model = AutoAdapterModel.from_pretrained(base)

        if load_proximity and proximity is not None:
            log.info("Loading proximity adapter from %s (corpus / title+abstract)", proximity)
            self.model.load_adapter(str(proximity), load_as=_ADAPTER_PROXIMITY, set_active=False)

        if load_adhoc and adhoc is not None:
            log.info("Loading adhoc adapter from %s (retrieval queries)", adhoc)
            self.model.load_adapter(str(adhoc), load_as=_ADAPTER_ADHOC, set_active=False)

        if adapter_checkpoint is not None:
            self._load_finetuned_adapters(Path(adapter_checkpoint))

        if load_proximity:
            self.model.set_active_adapters(_ADAPTER_PROXIMITY)
        elif load_adhoc:
            self.model.set_active_adapters(_ADAPTER_ADHOC)

        self.model.to(self.device)
        self.model.eval()

        if load_proximity:
            self.embed_corpus(["."])
        elif load_adhoc:
            self.embed_query(["."])

    def _load_finetuned_adapters(self, checkpoint_dir: Path) -> None:
        """Load ``adapter_adhoc`` / ``adapter_proximity`` saved by ``specter2_contrastive_train.py``."""
        for name, load_as in (
            ("adapter_adhoc", _ADAPTER_ADHOC),
            ("adapter_proximity", _ADAPTER_PROXIMITY),
        ):
            sub = checkpoint_dir / name
            if not sub.is_dir():
                continue
            log.info("Loading fine-tuned adapter from %s as %r", sub, load_as)
            self.model.load_adapter(str(sub), load_as=load_as, set_active=False)

    def _ensure_adapter(self, adapter: _AdapterKey) -> None:
        if adapter == _ADAPTER_PROXIMITY and self.adapters not in {"corpus", "both"}:
            raise RuntimeError("Proximity adapter is not loaded. Create embedder with adapters='corpus' or 'both'.")
        if adapter == _ADAPTER_ADHOC and self.adapters not in {"query", "both"}:
            raise RuntimeError("Adhoc adapter is not loaded. Create embedder with adapters='query' or 'both'.")

    @torch.inference_mode()
    def _encode_cls(self, texts: list[str], adapter: _AdapterKey) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_adapter(adapter)
        self.model.set_active_adapters(adapter)
        log.info("Encoding with adapter %r (%d chunk(s))", adapter, len(texts))

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_embed_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        return cls.tolist()

    @torch.inference_mode()
    def _embed_one(self, text: str, adapter: _AdapterKey) -> list[float]:
        """Embed one document; mean-pool CLS vectors when text exceeds ``max_embed_tokens``."""
        _token_count, chunks = split_text_for_embedding(
            self.tokenizer,
            text,
            max_length=self.max_embed_tokens,
            stride=self.chunk_stride,
        )
        if len(chunks) == 1:
            return self._encode_cls(chunks, adapter)[0]
        chunk_vectors = np.array(self._encode_cls(chunks, adapter), dtype=np.float32)
        return chunk_vectors.mean(axis=0).tolist()

    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        """Document vectors for title+abstract: **proximity** adapter only."""
        return [self._embed_one(text, _ADAPTER_PROXIMITY) for text in texts]

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Query vectors at retrieval time: **adhoc_query** adapter only."""
        return self._encode_cls(texts, _ADAPTER_ADHOC)
