"""
graphrag/tool.py - Agent tool definition for GraphRAG repository retrieval.

Exposes GraphRAG retrieval as a LangChain StructuredTool for use in planning
and build subagents.

Fixes (B6):
- project_id is bound in the tool factory from the caller's session context;
  the LLM can no longer override or omit it so cross-tenant leakage is prevented.
- language defaults to None (not 'python') so JS/TS projects are not silently
  excluded when the model forgets to specify the language.
"""

from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from graphrag.pipeline import GraphRAGPipeline


class _GraphRAGQueryInput(BaseModel):
    query: str = Field(
        description="The technical question or todo item describing what code to retrieve."
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "Optional programming language filter (e.g. 'python', 'javascript', 'typescript'). "
            "Leave unset to search all languages in the project."
        ),
    )
    module_scope: Optional[str] = Field(
        default=None,
        description="Optional file path prefix to scope search (e.g. 'billing/' or 'auth/').",
    )
    # B6: project_id is NOT exposed to the model; it is bound server-side via create_graphrag_retriever_tool.


def create_graphrag_retriever_tool(
    pipeline: Optional[GraphRAGPipeline] = None,
    project_id: Optional[str] = None,
) -> StructuredTool:
    """Creates a StructuredTool wrapping GraphRAG retrieval with grounded citations.

    Args:
        pipeline: Shared GraphRAGPipeline instance (B7: must be the same object used by ingest).
        project_id: Tenant project identifier bound server-side so the model cannot
            override or omit it (B6 fix).  Pass None only in dev/test environments.
    """
    _pipeline = pipeline or GraphRAGPipeline()
    _project_id = project_id  # bound at factory time; not accepted from the model (B6)

    def _retrieve_graphrag_code(
        query: str,
        language: Optional[str] = None,
        module_scope: Optional[str] = None,
    ) -> str:
        """Queries the Neo4j GraphRAG store and returns grounded code chunks with [path:line] citations."""
        return _pipeline.assemble_retrieval_context(
            query=query,
            project_id=_project_id,  # always use server-bound project_id (B6)
            language=language,
            module_scope=module_scope,
        )

    return StructuredTool.from_function(
        func=_retrieve_graphrag_code,
        name="retrieve_graphrag_code",
        description=(
            "Retrieves relevant code units (functions, classes) from the repository's AST graph "
            "and vector index with grounded [path:line] citations. Eliminates hallucinated code references. "
            "project_id is bound server-side and must NOT be passed by the model."
        ),
        args_schema=_GraphRAGQueryInput,
    )


# ---------------------------------------------------------------------------
# Neo4j MCP Server direct integration for GraphRAG
# ---------------------------------------------------------------------------
from frameworks.mcp_client import Neo4jMCPClient, neo4j_mcp_client


async def create_neo4j_graphrag_mcp_tools(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    read_only: bool = True,  # B6: default to read-only; never give tenant agents raw Cypher writes
    command: str = "python",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[list[StructuredTool], Neo4jMCPClient]:
    """Connects to the Neo4j MCP server (python -m neo4j_mcp_server) via stdio

    and exposes its direct Cypher read/write and schema inspection tools for GraphRAG.

    Returns:
        (tools, client_instance) - Keep client_instance to close on teardown.

    Note (B6): read_only defaults to True.  Do NOT pass read_only=False to any
    tenant-facing agent; doing so grants raw Cypher write access that bypasses
    project_id isolation.
    """
    client = neo4j_mcp_client(
        uri=uri,
        username=username,
        password=password,
        database=database,
        read_only=read_only,
        command=command,
        args=args,
        env=env,
    )
    tools = await client.get_tools()
    return tools, client
