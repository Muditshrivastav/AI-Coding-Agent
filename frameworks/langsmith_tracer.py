"""
frameworks/langsmith_tracer.py - LangSmith observability and tracing integration.
"""

import os
from typing import Any
from langsmith import Client, evaluate, traceable


class LangSmithTracer:
    """Wraps LangSmith tracing client and provides runtime tracing callbacks."""

    def __init__(self, project_name: str | None = None) -> None:
        self._project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "coding-agent-harness")
        self._client = Client()

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def client(self) -> Client:
        return self._client

    def get_run_url(self, run_id: str) -> str:
        return f"https://smith.langchain.com/o/default/projects/p/{self._project_name}/r/{run_id}"
