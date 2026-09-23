"""
graphrag/embedder.py - Embedding generation abstraction.

Provides embeddings for AST code chunks and search queries, with support for
HuggingFace / Ollama / local sentence-transformers models, and an offline hash-based fallback.
"""

from abc import ABC, abstractmethod
from typing import Sequence
import hashlib
import math


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
    Uses langchain-huggingface HuggingFaceEmbeddings with Qwen/Qwen3-Embedding-0.6B.
    If unavailable or offline, computes deterministic normalized pseudo-embeddings
    to ensure tests and local environments run reliably without external downloads.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dimension: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._dim = dimension
        self._hf_embeddings = None

        # Attempt to load langchain-huggingface HuggingFaceEmbeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
            )
            # Detect dimension if possible
            sample_vec = self._hf_embeddings.embed_query("test")
            self._dim = len(sample_vec)
        except Exception:
            # Fallback to sentence_transformers directly if available
            try:
                from sentence_transformers import SentenceTransformer

                st_model = SentenceTransformer(model_name, trust_remote_code=True)
                self._dim = st_model.get_sentence_embedding_dimension()
                self._hf_embeddings = st_model
            except Exception:
                self._hf_embeddings = None

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        if self._hf_embeddings is not None:
            try:
                if hasattr(self._hf_embeddings, "embed_query"):
                    return [float(x) for x in self._hf_embeddings.embed_query(text)]
                elif hasattr(self._hf_embeddings, "encode"):
                    emb = self._hf_embeddings.encode(text, convert_to_numpy=True)
                    return emb.tolist()
            except Exception:
                pass
        return self._deterministic_hash_vector(text, self._dim)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if self._hf_embeddings is not None:
            try:
                if hasattr(self._hf_embeddings, "embed_documents"):
                    return [[float(x) for x in doc] for doc in self._hf_embeddings.embed_documents(list(texts))]
                elif hasattr(self._hf_embeddings, "encode"):
                    embs = self._hf_embeddings.encode(list(texts), convert_to_numpy=True)
                    return embs.tolist()
            except Exception:
                pass
        return [self._deterministic_hash_vector(t, self._dim) for t in texts]

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

