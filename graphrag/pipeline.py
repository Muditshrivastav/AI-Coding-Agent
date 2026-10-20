"""
graphrag/pipeline.py - Full GraphRAG pipeline orchestrator.

Coordinates ingestion (GitHub via LlamaIndex or local checkout -> AST parser -> Neo4j store),
metadata-filtered vector search, graph-hop expansion, cross-encoder reranking, faithfulness
context assembly, and the anti-hallucination retry loop.

Fixes applied:
  B1 - retrieve_and_check() wires retrieve -> generate -> evaluate -> narrow_query -> retry.
  B2 - retrieve() calls store.expand_neighbors() to add CALLS/INHERITS hop context.
  B3/B4 - ingest methods call store.resolve_edges() as a second pass after storing nodes.
  B8 - init_schema() is called with the live embedder dimension so the vector index is
       created with the correct size.
"""

from typing import Any, Callable
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
        self.embedder = embedder or DefaultEmbedder(
            model_name=self.config.embedding_model_name,
            dimension=self.config.embedding_dimension,
        )
        # B8: pass the real dimension from the embedder so the vector index is sized correctly.
        self.store = store or Neo4jGraphStore(self.config)
        if self.store.is_connected():
            self.store.init_schema(embedding_dimension=self.embedder.dimension)
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

        Two-phase: store_chunks (nodes only) then resolve_edges (B3/B4).
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

        # Embed source text only (docstring + signature embedding improvement noted as future work)
        sources = [c.source for c in chunks]
        embeddings = self.embedder.embed_batch(sources)

        # Phase 1: store nodes
        count = self.store.store_chunks(chunks, embeddings)
        logger.info(f"Stored {count} code chunk nodes for project '{pid}'.")

        # Phase 2: resolve edges (B3/B4: two-pass so forward refs always exist)
        self.store.resolve_edges(chunks)
        logger.info(f"Resolved structural edges for project '{pid}'.")

        return count

    def ingest_local_repo(
        self,
        directory_path: str,
        project_id: str = "default",
        commit_sha: str = "",
    ) -> int:
        """
        Ingests a local project checkout using AST code-aware chunking and Neo4j storage.

        Two-phase: store_chunks (nodes only) then resolve_edges (B3/B4).
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

        # Phase 2: resolve edges (B3/B4)
        self.store.resolve_edges(chunks)

        return count

    def retrieve(
        self,
        query: str,
        project_id: str | None = None,
        language: str | None = None,
        module_scope: str | None = None,
        top_k: int | None = None,
        expand_graph: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieval pipeline:
        1. Embed the query.
        2. Vector search in Neo4j with Layer 3 metadata filtering.
        3. Rerank top candidates with CrossEncoder.
        4. (B2) Expand 1-hop CALLS/INHERITS neighbours from top-ranked hits.
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

        # 4. B2: expand graph neighbours from top hits so callers/callees/bases are included
        if expand_graph and project_id:
            neighbours = self.store.expand_neighbors(
                hit_nodes=reranked,
                project_id=project_id,
                neighbor_cap=5,
            )
            # Append neighbour nodes that aren't already in the ranked list
            reranked = reranked + neighbours

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

    def retrieve_and_check(
        self,
        query: str,
        generate_fn: Callable[[str, str], str],
        project_id: str | None = None,
        language: str | None = None,
        module_scope: str | None = None,
    ) -> dict[str, Any]:
        """
        B1: Wires the full anti-hallucination loop:
          retrieve -> format_context -> generate -> evaluate_faithfulness
          -> if failed: narrow_query -> retry up to max_faithfulness_retries.

        Args:
            query: Natural-language question or todo item.
            generate_fn: Callable(context_str, query_str) -> answer_str.
                         Typically wraps your LLM call.
            project_id: Tenant scope for retrieval.
            language: Optional language filter.
            module_scope: Optional file-path prefix filter.

        Returns:
            dict with keys: answer, score, passed, retries, context.
        """
        current_query = query
        best_result: dict[str, Any] = {}

        for attempt in range(self.config.max_faithfulness_retries + 1):
            candidates = self.retrieve(
                query=current_query,
                project_id=project_id,
                language=language,
                module_scope=module_scope,
            )
            context_str = self.faithfulness.format_retrieval_context(candidates)
            context_texts = [c.get("source", "") for c in candidates]

            answer = generate_fn(context_str, current_query)

            eval_result = self.faithfulness.evaluate_faithfulness(
                query=current_query,
                retrieved_contexts=context_texts,
                generated_answer=answer,
            )

            best_result = {
                "answer": answer,
                "score": eval_result["score"],
                "passed": eval_result["passed"],
                "retries": attempt,
                "context": context_str,
                "unsupported_claims": eval_result.get("unsupported_claims", []),
            }

            if eval_result["passed"]:
                logger.info(
                    f"[pipeline] Faithfulness check passed on attempt {attempt + 1} "
                    f"(score={eval_result['score']:.3f})."
                )
                break

            if attempt < self.config.max_faithfulness_retries:
                unsupported = eval_result.get("unsupported_claims", [])
                current_query = self.faithfulness.narrow_query(current_query, unsupported)
                logger.info(
                    f"[pipeline] Faithfulness check failed (score={eval_result['score']:.3f}, "
                    f"unsupported={unsupported}). Retrying with narrowed query: '{current_query}'"
                )
            else:
                logger.warning(
                    f"[pipeline] Faithfulness loop exhausted after {attempt + 1} attempts. "
                    f"Final score={eval_result['score']:.3f}. "
                    "Returning best available answer."
                )

        return best_result
