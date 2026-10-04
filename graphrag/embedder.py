"""
graphrag/embedder.py - Embedding generation abstraction.

Provides embeddings for AST code chunks and search queries, with support for
HuggingFace / Ollama / local sentence-transformers models, and an offline hash-based fallback.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Any, Union
import hashlib
import math
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    If unavailable or offline, computes deterministic normalized pseudo-embeddings
    to ensure tests and local environments run reliably without external downloads.
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:4b",
        dimension: int = 4096,
    ) -> None:
        self._model_name = model_name
        self._dim = dimension
        self._ollama_embeddings: Union["OllamaEmbeddings", "SentenceTransformer", None] = None

        # Attempt to load langchain-ollama OllamaEmbeddings
        try:
            from langchain_ollama import OllamaEmbeddings

            # Strip any accidental whitespace from model name before passing to Ollama
            self._ollama_embeddings = OllamaEmbeddings(
                model=model_name.strip(),
            )
            # Detect dimension if possible
            sample_vec = self._ollama_embeddings.embed_query("test")
            self._dim = len(sample_vec)
        except Exception:
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
                except Exception:
                    self._ollama_embeddings = None
            else:
                
                self.ollama_embeddings = None

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
            except Exception:
                pass
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
            except Exception:
                pass
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

