"""
tools/codebase_retriever.py - Embeddings and context retriever over target codebase.
"""

from typing import Any
from pathlib import Path
from langchain_core.tools import StructuredTool


class CodebaseRetriever:
    """Retrieves relevant code segments and symbols from the local codebase."""

    def __init__(self, root_dir: str = ".") -> None:
        self._root_dir = Path(root_dir)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Walks repository and finds matches for the query."""
        results = []
        for p in self._root_dir.rglob("*.py"):
            if ".venv" in p.parts or "__pycache__" in p.parts:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if any(q.lower() in content.lower() for q in query.split()):
                    results.append(
                        {
                            "source": "codebase",
                            "file": str(p.relative_to(self._root_dir)),
                            "content": content[:1500],
                        }
                    )
                    if len(results) >= top_k:
                        break
            except Exception:
                continue
        return results

    def as_tool(self) -> StructuredTool:
        def _search_codebase(query: str) -> str:
            matches = self.retrieve(query)
            if not matches:
                return f"No code matches found for '{query}'."
            return "\n\n".join(f"[{m['file']}]:\n{m['content']}" for m in matches)

        return StructuredTool.from_function(
            func=_search_codebase,
            name="retrieve_codebase_context",
            description="Searches local project codebase for relevant functions, classes, and logic.",
        )
