"""
graphrag/tool.py - Agent tool definition for GraphRAG repository retrieval.

Exposes GraphRAG retrieval as a LangChain StructuredTool for use in planning
and build subagents.
"""

from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from graphrag.pipeline import GraphRAGPipeline


class _GraphRAGQueryInput(BaseModel):
    query: str = Field(
        description="The technical question or todo item describing what code to retrieve."
    )
    project_id: Optional[str] = Field(
        default=None,
        description="Optional project identifier (e.g. 'owner/repo' or 'default') to isolate search.",
    )
    language: Optional[str] = Field(
        default="python",
        description="Programming language filter (e.g. 'python', 'javascript').",
    )
    module_scope: Optional[str] = Field(
        default=None,
        description="Optional file path prefix to scope search (e.g. 'billing/' or 'auth/').",
    )


def create_graphrag_retriever_tool(
    pipeline: Optional[GraphRAGPipeline] = None,
) -> StructuredTool:
    """Creates a StructuredTool wrapping GraphRAG retrieval with grounded citations."""
    _pipeline = pipeline or GraphRAGPipeline()

    def _retrieve_graphrag_code(
        query: str,
        project_id: Optional[str] = None,
        language: Optional[str] = "python",
        module_scope: Optional[str] = None,
    ) -> str:
        """Queries the Neo4j GraphRAG store and returns grounded code chunks with [path:line] citations."""
        return _pipeline.assemble_retrieval_context(
            query=query,
            project_id=project_id,
            language=language,
            module_scope=module_scope,
        )

    return StructuredTool.from_function(
        func=_retrieve_graphrag_code,
        name="retrieve_graphrag_code",
        description=(
            "Retrieves relevant code units (functions, classes) from the repository's AST graph "
            "and vector index with grounded [path:line] citations. Eliminates hallucinated code references."
        ),
        args_schema=_GraphRAGQueryInput,
    )
