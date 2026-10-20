"""
graphrag/embedder.py - Embedding generation abstraction.

Provides embeddings for AST code chunks and search queries, with support for
HuggingFace / Ollama / local sentence-transformers models, and an offline hash-based fallback.

Fixes (B8):
- Corrected typo: was `self.ollama_embeddings = None` (public attr); now always `self._ollama_embeddings = None`.
- Dimension is auto-detected from the live model on init instead of relying on the hard-coded config value.
- Active embedder (Ollama, SentenceTransformer, or hash) is logged at INFO level so mismatches are visible.
- allow_fallback=False (production default) raises instead of silently producing meaningless hash vectors.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Any, Union
import hashlib
import logging
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for vector embedders."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embeds a single text snippet into a vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embeds multiple text snippets into vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        pass


class DefaultEmbedder(BaseEmbedder):
    """
    Uses langchain-ollama OllamaEmbeddings with Qwen3-Embedding-4B.

    If unavailable or offline and allow_fallback=True, computes deterministic
    normalized pseudo-embeddings so tests and local environments run without
    external services.  In production (allow_fallback=False) the constructor
    raises so a mis-configured embedder surfaces at startup rather than
    silently producing meaningless vectors that corrupt the index.

    B8 fixes applied:
    - Typo corrected: self._ollama_embeddings is the sole attribute name.
    - Dimension is probed from the live model; the config default is a last resort.
    - Which embedder is active is logged at INFO level.
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:4b",
        dimension: int = 4096,
        allow_fallback: bool = True,
    ) -> None:
        self._model_name = model_name
        # _dim will be overwritten once we probe the live model (B8: don't trust config hard-code).
        self._dim: int = dimension
        # B8 fix: always use self._ollama_embeddings (was accidentally self.ollama_embeddings in one branch).
        self._ollama_embeddings: Union["OllamaEmbeddings", "SentenceTransformer", None] = None
        self._allow_fallback = allow_fallback

        # Attempt to load langchain-ollama OllamaEmbeddings
        try:
            from langchain_ollama import OllamaEmbeddings

            embedder = OllamaEmbeddings(model=model_name.strip())
            # B8: probe real dimension from the live model instead of trusting the config value.
            sample_vec = embedder.embed_query("test")
            self._dim = len(sample_vec)
            self._ollama_embeddings = embedder
            logger.info(
                f"[embedder] Active: OllamaEmbeddings(model='{model_name}'), dim={self._dim}"
            )
            return
        except Exception as ollama_exc:
            logger.debug(f"[embedder] Ollama not available ({ollama_exc}), trying SentenceTransformer.")

        # Fallback to sentence_transformers — ONLY for HuggingFace-style model names.
        # Never attempt a download for Ollama tags (e.g. "qwen3-embedding:4b") since
        # SentenceTransformer would try to fetch from HuggingFace Hub and loop forever.
        _looks_like_hf = "/" in model_name and ":" not in model_name
        if _looks_like_hf:
            try:
                from sentence_transformers import SentenceTransformer

                st_model = SentenceTransformer(model_name.strip(), trust_remote_code=True)
                self._dim = st_model.get_embedding_dimension()
                self._ollama_embeddings = st_model
                logger.info(
                    f"[embedder] Active: SentenceTransformer(model='{model_name}'), dim={self._dim}"
                )
                return
            except Exception as st_exc:
                logger.debug(f"[embedder] SentenceTransformer not available ({st_exc}).")

        # No real embedder available.
        self._ollama_embeddings = None
        if allow_fallback:
            logger.warning(
                f"[embedder] No embedding model available for '{model_name}'. "
                "Falling back to deterministic hash-based pseudo-vectors. "
                "Ingest and retrieval will 'work' but return MEANINGLESS results. "
                "Fix: ensure Ollama is running or set a valid HuggingFace model_name."
            )
        else:
            raise RuntimeError(
                f"[embedder] No embedding model available for '{model_name}' "
                "and allow_fallback=False. Ensure Ollama is running or provide a valid "
                "HuggingFace model name."
            )

    @property
    def dimension(self) -> int:
        return self._dim if self._dim is not None else 4096

    def embed_text(self, text: str) -> list[float]:
        if self._ollama_embeddings is not None:
            try:
                # Cast to Any to avoid Pylance confusion between Ollama and SentenceTransformer APIs
                model: Any = self._ollama_embeddings
                if hasattr(model, "embed_query"):
                    return [float(x) for x in model.embed_query(text)]
                elif hasattr(model, "encode"):
                    emb = model.encode(text, convert_to_numpy=True)
                    return emb.tolist()
            except Exception as exc:
                logger.warning(f"[embedder] embed_text failed ({exc}), falling back to hash vector.")
        if not self._allow_fallback:
            raise RuntimeError("[embedder] embed_text: real embedder failed and allow_fallback=False.")
        return self._deterministic_hash_vector(text, self._dim or 4096)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._ollama_embeddings is not None:
            try:
                # Cast to Any to avoid Pylance confusion between Ollama and SentenceTransformer APIs
                model: Any = self._ollama_embeddings
                if hasattr(model, "embed_documents"):
                    return [[float(x) for x in doc] for doc in model.embed_documents(list(texts))]
                elif hasattr(model, "encode"):
                    embs = model.encode(list(texts), convert_to_numpy=True)
                    return embs.tolist()
            except Exception as exc:
                logger.warning(f"[embedder] embed_batch failed ({exc}), falling back to hash vectors.")
        if not self._allow_fallback:
            raise RuntimeError("[embedder] embed_batch: real embedder failed and allow_fallback=False.")
        return [self._deterministic_hash_vector(t, self._dim or 4096) for t in texts]

    @staticmethod
    def _deterministic_hash_vector(text: str, dim: int) -> list[float]:
        """Produces a deterministic unit-length vector from text tokens for local fallback."""
        vec = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec

        for idx, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            pos = h % dim
            weight = 1.0 / (1.0 + math.log(idx + 1))
            vec[pos] += weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec
