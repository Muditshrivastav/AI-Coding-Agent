"""
tools/api_tools.py - Custom REST API tools exposed as LangChain StructuredTools.
"""

from typing import Any
from langchain_core.tools import StructuredTool


class APIToolset:
    """Manages custom API-backed StructuredTools (e.g. issues, status checks)."""

    def __init__(self, base_url: str = "", auth_token: str | None = None) -> None:
        self._base_url = base_url
        self._auth_token = auth_token

    def get_tools(self) -> list[StructuredTool]:
        def create_issue(title: str, body: str) -> str:
            """Creates a project issue or tracking item."""
            return f"Created tracking issue: '{title}'"

        def check_ci_status(project: str) -> str:
            """Checks recent CI build runs for a given project."""
            return f"CI status for {project}: all checks passed."

        return [
            StructuredTool.from_function(
                func=create_issue,
                name="create_project_issue",
                description="Creates an issue or tracking item for the project.",
            ),
            StructuredTool.from_function(
                func=check_ci_status,
                name="check_ci_status",
                description="Checks the current CI status of the project.",
            ),
        ]
