"""
graphrag/ingest_tool.py - Agent tool for automatic GitHub repo ingestion into Neo4j.

Exposes `ingest_github_repo` as a LangChain StructuredTool so the agent can
automatically index any GitHub repository into the Neo4j GraphRAG store whenever
the user mentions a repo URL or owner/repo reference.
"""

import re
import logging
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from graphrag.pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)

# Lazy singleton – created once on first tool call to avoid heavy startup costs.
_pipeline: Optional[GraphRAGPipeline] = None


def _get_pipeline() -> GraphRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphRAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class _IngestRepoInput(BaseModel):
    repo_url: str = Field(
        description=(
            "GitHub repository to ingest. Accepts any of the following formats:\n"
            "  - Full URL:  https://github.com/owner/repo\n"
            "  - Short ref: owner/repo\n"
            "Example: 'openai/openai-python' or 'https://github.com/openai/openai-python'"
        )
    )
    branch: str = Field(
        default="main",
        description="Branch to ingest (default: 'main'). Use 'master' for older repos.",
    )


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------

_GITHUB_URL_RE = re.compile(
    r"(?:https?://github\.com/)?(?P<owner>[^/\s]+)/(?P<repo>[^/\s#?]+)"
)


def _parse_repo(repo_url: str) -> tuple[str, str]:
    """Parse 'owner/repo' or full GitHub URL into (owner, repo) pair."""
    match = _GITHUB_URL_RE.search(repo_url.strip())
    if not match:
        raise ValueError(
            f"Cannot parse GitHub repo from '{repo_url}'. "
            "Expected format: 'owner/repo' or 'https://github.com/owner/repo'."
        )
    return match.group("owner"), match.group("repo")


def _ingest_github_repo(repo_url: str, branch: str = "main") -> str:
    """
    Ingests a GitHub repository into the Neo4j GraphRAG store.

    Parses the repo reference, pulls all source files via the GitHub API,
    runs AST-based code chunking, embeds each chunk, and stores nodes +
    relationships in Neo4j so the agent can retrieve them with
    `retrieve_graphrag_code`.

    Returns a human-readable status string with the number of chunks stored.
    """
    try:
        owner, repo = _parse_repo(repo_url)
    except ValueError as exc:
        return str(exc)

    project_id = f"{owner}/{repo}"
    logger.info(f"[ingest_tool] Ingesting {project_id} (branch: {branch})...")

    try:
        pipeline = _get_pipeline()
        count = pipeline.ingest_github_repo(
            owner=owner,
            repo=repo,
            branch=branch,
            project_id=project_id,
        )
        if count == 0:
            return (
                f"Repository '{project_id}' was ingested but no parseable code chunks "
                "were found. Check that the repo is non-empty and GITHUB_TOKEN is valid."
            )
        return (
            f"Successfully ingested '{project_id}' (branch: '{branch}') into Neo4j. "
            f"{count} code chunks stored. You can now query this repo with "
            f"retrieve_graphrag_code(project_id='{project_id}')."
        )
    except RuntimeError as exc:
        logger.error(f"[ingest_tool] Ingestion failed for {project_id}: {exc}")
        return (
            f"Ingestion failed for '{project_id}': {exc}. "
            "Ensure GITHUB_TOKEN is set in .env and Neo4j is running."
        )
    except Exception as exc:
        logger.error(f"[ingest_tool] Unexpected error for {project_id}: {exc}")
        return f"Unexpected error while ingesting '{project_id}': {exc}"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_repo_ingest_tool(
    pipeline: Optional[GraphRAGPipeline] = None,
) -> StructuredTool:
    """
    Returns a StructuredTool that the agent can call to automatically index a
    GitHub repository into Neo4j whenever the user mentions a repo.

    Args:
        pipeline: Optional pre-constructed GraphRAGPipeline. If None, a singleton
                  is created lazily on first tool call.
    """
    if pipeline is not None:
        # Override the module-level singleton with the injected pipeline.
        global _pipeline
        _pipeline = pipeline

    return StructuredTool.from_function(
        func=_ingest_github_repo,
        name="ingest_github_repo",
        description=(
            "Indexes a GitHub repository into the Neo4j graph database so the agent "
            "can later retrieve code from it. Call this automatically whenever the user "
            "provides a GitHub URL or owner/repo reference that has not been ingested yet. "
            "After ingestion, use retrieve_graphrag_code with the same project_id to query it."
        ),
        args_schema=_IngestRepoInput,
    )
