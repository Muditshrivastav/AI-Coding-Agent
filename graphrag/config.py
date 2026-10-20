"""
graphrag/config.py - Configuration definitions for GraphRAG pipelines.
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


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
    embedding_dimension: int = 4096  # overridden by live embedder probe in pipeline init (B8)
    embedding_model_name: str = "qwen3-embedding:4b"
    reranker_model_name: str = "BAAI/bge-reranker-base"
    vector_search_top_k: int = 20
    reranker_top_k: int = 5

    # Faithfulness & Guardrails Settings
    faithfulness_threshold: float = 0.8
    max_faithfulness_retries: int = 2

    def __post_init__(self) -> None:
        # Warn loudly if the password is still the dev default in a non-test environment.
        if self.neo4j_password == "password" and not os.getenv("PYTEST_CURRENT_TEST"):
            logger.warning(
                "[config] NEO4J_PASSWORD is still the default 'password'. "
                "Set NEO4J_PASSWORD in .env or the environment before deploying."
            )
