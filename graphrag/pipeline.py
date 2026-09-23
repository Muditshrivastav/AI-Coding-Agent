"""
graphrag/pipeline.py - Full GraphRAG pipeline orchestrator.

Coordinates ingestion (GitHub via LlamaIndex or local checkout -> AST parser -> Neo4j store),
metadata-filtered vector search, cross-encoder reranking, and faithfulness context assembly.
"""

from typing import Any
import logging

from graphrag.config import GraphRAGConfig
from graphrag.github_reader import GitHubRepoIngester
from graphrag.store import Neo4jGraphStore
from graphrag.embedder import BaseEmbedder, DefaultEmbedder
from graphrag.reranker import BaseReranker, CrossEncoderReranker
from graphrag.faithfulness import FaithfulnessScorer

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """End-to-end GraphRAG coordinator for GitHub repository code search and ingestion."""

    def __init__(
        self,
        config: GraphRAGConfig | None = None,
        store: Neo4jGraphStore | None = None,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        faithfulness_scorer: FaithfulnessScorer | None = None,
    ) -> None:
        self.config = config or GraphRAGConfig()
        self.store = store or Neo4jGraphStore(self.config)
        self.embedder = embedder or DefaultEmbedder(
            model_name=self.config.embedding_model_name,
            dimension=self.config.embedding_dimension,
        )
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranker_model_name
        )
        self.faithfulness = faithfulness_scorer or FaithfulnessScorer(
            threshold=self.config.faithfulness_threshold
        )
        self.ingester = GitHubRepoIngester(github_token=self.config.github_token)

    def ingest_github_repo(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        project_id: str | None = None,
        commit_sha: str = "",
    ) -> int:
        """
        Ingests a GitHub repository via LlamaIndex GithubRepositoryReader (use_parser=False),
        parses symbols via AST, embeds them, and inserts nodes + relationships into Neo4j.
        """
        pid = project_id or f"{owner}/{repo}"
        logger.info(f"Ingesting GitHub repo {owner}/{repo} (branch: {branch})...")
        chunks = self.ingester.ingest_from_github(
            owner=owner,
            repo=repo,
            branch=branch,
            project_id=pid,
            commit_sha=commit_sha,
        )
        if not chunks:
            logger.warning(f"No parseable code chunks found in repo {owner}/{repo}.")
            return 0

        sources = [c.source for c in chunks]
        embeddings = self.embedder.embed_batch(sources)
        count = self.store.store_chunks(chunks, embeddings)
        logger.info(f"Stored {count} code chunks in graph for project '{pid}'.")
        return count

    def ingest_local_repo(
        self,
        directory_path: str,
        project_id: str = "default",
        commit_sha: str = "",
    ) -> int:
        """
        Ingests a local project checkout using AST code-aware chunking and Neo4j storage.
        """
        logger.info(f"Ingesting local directory {directory_path} (project_id: {project_id})...")
        chunks = self.ingester.ingest_from_local_directory(
            directory_path=directory_path,
            project_id=project_id,
            commit_sha=commit_sha,
        )
        if not chunks:
            return 0

        sources = [c.source for c in chunks]
        embeddings = self.embedder.embed_batch(sources)
        count = self.store.store_chunks(chunks, embeddings)
        return count

    def retrieve(
        self,
        query: str,
        project_id: str | None = None,
        language: str | None = None,
        module_scope: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieval pipeline:
        1. Embed the query.
        2. Vector search in Neo4j with Layer 3 metadata filtering.
        3. Rerank top candidates with CrossEncoder.
        """
        top_k = top_k or self.config.reranker_top_k
        vector_candidates_k = self.config.vector_search_top_k

        # 1. Embed query
        query_emb = self.embedder.embed_text(query)

        # 2. Vector search with metadata filters (Layer 3)
        candidates = self.store.query_similar_nodes(
            query_embedding=query_emb,
            project_id=project_id,
            language=language,
            module_scope=module_scope,
            top_k=vector_candidates_k,
        )

        if not candidates:
            return []

        # 3. Rerank candidates with CrossEncoder
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return reranked

    def assemble_retrieval_context(
        self,
        query: str,
        project_id: str | None = None,
        language: str | None = None,
        module_scope: str | None = None,
    ) -> str:
        """Retrieves and formats code chunks into citation-enforced context string."""
        candidates = self.retrieve(
            query=query,
            project_id=project_id,
            language=language,
            module_scope=module_scope,
        )
        return self.faithfulness.format_retrieval_context(candidates)
