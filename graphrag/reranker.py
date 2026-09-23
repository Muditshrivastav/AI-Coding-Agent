"""
graphrag/reranker.py - Cross-encoder reranker.

Performs a second, more accurate relevance pass over top vector candidates.
Uses BAAI/bge-reranker-base (or fallback cross-scoring) to eliminate superficial
vector matches and surface only genuinely relevant code.
"""

from abc import ABC, abstractmethod
from typing import Any
import math


class BaseReranker(ABC):
    """Abstract base class for candidate rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Reranks candidate dictionary items by relevance to the query."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Reranks candidates using CrossEncoder('BAAI/bge-reranker-base').
    Falls back gracefully to term-overlap lexical-boosted scoring if model is not loaded.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if self._model is not None:
            try:
                pairs = [(query, c.get("source", "")) for c in candidates]
                scores = self._model.predict(pairs)
                ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
                results = []
                for cand, score in ranked[:top_k]:
                    item = dict(cand)
                    item["rerank_score"] = float(score)
                    results.append(item)
                return results
            except Exception:
                pass

        # Fallback reranker combining vector score and token overlap
        return self._fallback_rerank(query, candidates, top_k)

    def _fallback_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_words = set(query.lower().split())
        scored = []

        for c in candidates:
            source = c.get("source", "").lower()
            name = c.get("name", "").lower()
            overlap = sum(1.0 for w in query_words if w in source or w in name)
            base_score = float(c.get("score", 0.0))
            # Boost score with exact symbol or token matches
            combined = base_score + (overlap * 0.15)
            item = dict(c)
            item["rerank_score"] = combined
            scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
