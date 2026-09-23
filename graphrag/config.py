"""
graphrag/config.py - Configuration definitions for GraphRAG pipelines.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GraphRAGConfig:
    """GraphRAG system parameters, connection strings, and threshold configurations."""

    # Neo4j Settings
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # GitHub Settings
    github_token: str = field(
        default_factory=lambda: (
            os.getenv("GITHUB_TOKEN")
            or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
            or ""
        )
    )

    # Retrieval & Reranker Settings
    embedding_dimension: int = 1024
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    vector_search_top_k: int = 20
    reranker_top_k: int = 5

    # Faithfulness & Guardrails Settings
    faithfulness_threshold: float = 0.8
    max_faithfulness_retries: int = 2
