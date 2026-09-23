"""
graphrag/github_reader.py - GitHub repository ingestion using LlamaIndex.

Uses `llama-index-readers-github` (GithubRepositoryReader) with `use_parser=False`
to load whole repository files via GitHub API without arbitrary token-based chunking.
Provides fallback to local directory repository scanning for offline / test environments.
"""

from typing import Any
import os
from pathlib import Path

from graphrag.ast_parser import ASTCodeParser, CodeChunk


class GitHubRepoIngester:
    """Acquires files from GitHub using LlamaIndex GithubRepositoryReader and parses them via AST."""

    def __init__(
        self,
        github_token: str | None = None,
        filter_extensions: tuple[str, ...] = (".py", ".js", ".ts", ".md"),
        exclude_directories: tuple[str, ...] = (
            "node_modules",
            "dist",
            ".git",
            ".venv",
            "__pycache__",
        ),
    ) -> None:
        self.github_token = github_token or os.getenv("GITHUB_TOKEN", "")
        self.filter_extensions = filter_extensions
        self.exclude_directories = exclude_directories

    def ingest_from_github(
        self,
        owner: str,
        repo: str,
        branch: str = "main",
        project_id: str | None = None,
        commit_sha: str = "",
    ) -> list[CodeChunk]:
        """
        Pulls documents directly from GitHub using LlamaIndex GithubRepositoryReader,
        skipping LlamaIndex's internal token chunker (use_parser=False), and passes
        complete file texts to ASTCodeParser for code-aware semantic chunking.
        """
        project_id = project_id or f"{owner}/{repo}"
        parser = ASTCodeParser(project_id=project_id, commit_sha=commit_sha)

        try:
            # Attempt to use LlamaIndex GitHub reader
            from llama_index.readers.github import GithubRepositoryReader, GithubClient

            github_client = GithubClient(github_token=self.github_token, verbose=False)
            reader = GithubRepositoryReader(
                github_client=github_client,
                owner=owner,
                repo=repo,
                use_parser=False,  # CRITICAL: keep file whole so AST determines exact code boundaries
                filter_file_extensions=(
                    list(self.filter_extensions),
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
                filter_directories=(
                    list(self.exclude_directories),
                    GithubRepositoryReader.FilterType.EXCLUDE,
                ),
            )
            documents = reader.load_data(branch=branch)

            all_chunks: list[CodeChunk] = []
            for doc in documents:
                file_path = (
                    doc.metadata.get("file_path")
                    or doc.metadata.get("file_name")
                    or "unknown"
                )
                chunks = parser.parse_file(file_path=file_path, content=doc.text)
                all_chunks.extend(chunks)

            return all_chunks

        except (ImportError, Exception) as exc:
            # Graceful diagnostic or alternative ingestion if reader is unavailable or token fails
            raise RuntimeError(
                f"Failed to ingest from GitHub repo '{owner}/{repo}': {exc}. "
                "Ensure GITHUB_TOKEN is valid or use ingest_from_local_directory()."
            ) from exc

    def ingest_from_local_directory(
        self,
        directory_path: str,
        project_id: str = "default",
        commit_sha: str = "",
    ) -> list[CodeChunk]:
        """
        Scans a local repository checkout and extracts code-aware AST chunks.
        Used for local build sandboxes and offline verification.
        """
        root = Path(directory_path)
        if not root.exists():
            raise FileNotFoundError(f"Local repository path '{directory_path}' does not exist.")

        parser = ASTCodeParser(project_id=project_id, commit_sha=commit_sha)
        all_chunks: list[CodeChunk] = []

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            # Check excluded directories
            if any(part in self.exclude_directories for part in p.parts):
                continue

            # Check allowed extensions
            if p.suffix.lower() not in self.filter_extensions:
                continue

            try:
                rel_path = str(p.relative_to(root)).replace("\\", "/")
                content = p.read_text(encoding="utf-8", errors="ignore")
                chunks = parser.parse_file(file_path=rel_path, content=content)
                all_chunks.extend(chunks)
            except Exception:
                continue

        return all_chunks
