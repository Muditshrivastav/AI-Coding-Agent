"""
frameworks/mcp_client.py - Official MCP SDK client wrapper.
Wraps one MCP server (stdio or HTTP) and converts its tools into LangChain StructuredTools.
"""

import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any

import anyio
import httpx
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import TextContent
from pydantic import create_model

logger = logging.getLogger(__name__)


class MCPClientWrapper:
    """Wraps one MCP server (stdio or streamable HTTP) via the official MCP Python SDK

    and converts exposed tools into LangChain StructuredTools.
    """

    def __init__(self, transport: str, **kwargs: Any) -> None:
        self._transport = transport.lower()  # "stdio" | "http"
        self._kwargs = kwargs
        self._session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self, timeout: float = 10.0) -> None:
        if self._session is not None:
            return

        try:
            if self._transport == "stdio":
                params = StdioServerParameters(
                    command=self._kwargs["command"],
                    args=self._kwargs.get("args", []),
                    env=self._kwargs.get("env"),
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            elif self._transport == "http":
                http_client = None
                if self._kwargs.get("headers"):
                    http_client = await self._stack.enter_async_context(
                        create_mcp_http_client(headers=self._kwargs["headers"])
                    )
                conn = await self._stack.enter_async_context(
                    streamable_http_client(
                        self._kwargs["url"],
                        http_client=http_client,
                    )
                )
                # Both sse_client and streamable_http_client return (read, write) or (read, write, _)
                if isinstance(conn, (tuple, list)):
                    read, write = conn[0], conn[1]
                else:
                    read, write = conn.read, conn.write
            else:
                raise ValueError(
                    f"Unsupported transport: {self._transport}. Must be 'stdio' or 'http'."
                )

            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await asyncio.wait_for(self._session.initialize(), timeout=timeout)
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._session = None
        stack, self._stack = self._stack, AsyncExitStack()
        try:
            await stack.aclose()
        except Exception as exc:
            # Non-fatal: AsyncExitStack.aclose() still exits every registered
            # context manager even when one __aexit__ raises.  The remaining
            # resources are released; only the last raised exception bubbles up.
            logger.warning(
                "Non-fatal error during MCP %s transport cleanup "
                "(resources still released): %s: %s",
                self._transport,
                type(exc).__name__,
                exc,
            )

    def _make_tool(self, mcp_tool: Any) -> StructuredTool:
        schema = getattr(mcp_tool, "inputSchema", None) or {
            "type": "object",
            "properties": {},
        }
        fields: dict[str, Any] = {
            name: (str, ...) for name in schema.get("properties", {})
        }
        args_model = create_model(f"{mcp_tool.name}_Args", **fields) if fields else None

        async def _call(**kwargs: Any) -> str:
            if self._session is None:
                raise RuntimeError(
                    "MCPClientWrapper is not connected. Call connect() first."
                )
            result = await self._session.call_tool(mcp_tool.name, arguments=kwargs)
            return "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )

        return StructuredTool.from_function(
            coroutine=_call,
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            args_schema=args_model,
        )

    async def get_tools(self, timeout: float = 10.0) -> list[StructuredTool]:
        deadline = time.monotonic() + timeout
        try:
            if self._session is None:
                await self.connect(timeout=max(0.0, deadline - time.monotonic()))
            assert self._session is not None
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=max(0.0, deadline - time.monotonic()),
            )
            return [self._make_tool(t) for t in result.tools]
        except BaseException:
            await self.close()
            raise


# ---------------------------------------------------------------------------
# draw.io hosted MCP endpoint
# ---------------------------------------------------------------------------
DRAWIO_MCP_URL = "https://mcp.draw.io/mcp"

# JSON-RPC initialize payload as required by the MCP specification.
_MCP_INIT_PAYLOAD: dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "ai-coding-agent", "version": "1.0.0"},
    },
}


class DrawioMCPClient:
    """MCP client for the draw.io hosted endpoint (https://mcp.draw.io/mcp).

    Lifecycle::

        client = DrawioMCPClient()
        tools  = await client.get_tools()  # POSTs initialize, extracts session ID
        # tools is a list[StructuredTool] ready to pass to the design subagent
        await client.close()

    What it does internally:

    1. POSTs the MCP ``initialize`` JSON-RPC message to the draw.io endpoint
       via ``httpx.AsyncClient``.
    2. Reads the ``Mcp-Session-Id`` header from the response and stores it.
    3. Constructs an ``MCPClientWrapper(transport="http")`` with that session
       header so every subsequent MCP request is tied to the correct server
       session.
    4. Delegates ``get_tools()`` / ``close()`` to the inner wrapper.
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=300.0)
        self._session_id: str | None = None
        self._wrapper: MCPClientWrapper | None = None
        self._tool_names: list[str] = []

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    async def _initialize(self, timeout: float = 10.0) -> None:
        """POST the MCP initialize request and capture the Mcp-Session-Id header."""
        response = await asyncio.wait_for(
            self._http.post(
                DRAWIO_MCP_URL,
                json=_MCP_INIT_PAYLOAD,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            ),
            timeout=timeout,
        )
        response.raise_for_status()

        self._session_id = response.headers.get("Mcp-Session-Id")

        # Build session-scoped headers for all further MCP requests.
        session_headers: dict[str, str] = {}
        if self._session_id:
            session_headers["Mcp-Session-Id"] = self._session_id

        self._wrapper = MCPClientWrapper(
            transport="http",
            url=DRAWIO_MCP_URL,
            headers=session_headers,
        )

    async def get_tools(self, timeout: float = 10.0) -> list[StructuredTool]:
        """Initialize the draw.io session (if needed) and return LangChain tools."""
        if self._wrapper is None:
            await self._initialize(timeout)
        assert self._wrapper is not None
        tools = await self._wrapper.get_tools(timeout=timeout)
        self._tool_names = [tool.name for tool in tools]
        return tools

    async def close(self) -> None:
        """Tear down the MCP session and the underlying HTTP client."""
        if self._wrapper is not None:
            await self._wrapper.close()
            self._wrapper = None
        await self._http.aclose()


def drawio_mcp_client() -> DrawioMCPClient:
    """Return a DrawioMCPClient ready to connect to https://mcp.draw.io/mcp."""
    return DrawioMCPClient()


# ---------------------------------------------------------------------------
# Render hosted MCP endpoint
# ---------------------------------------------------------------------------
RENDER_MCP_URL = "https://mcp.render.com/mcp"


class RenderMCPClient:
    """MCP client for the Render hosted endpoint (https://mcp.render.com/mcp).

    Uses RENDER_API_KEY from environment or explicit argument.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("RENDER_API_KEY", "")
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=10.0)
        self._session_id: str | None = None
        self._wrapper: MCPClientWrapper | None = None

    async def _initialize(self, timeout: float = 10.0) -> None:
        """POST the MCP initialize request with Bearer authorization and capture session ID."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await asyncio.wait_for(
            self._http.post(
                RENDER_MCP_URL,
                json=_MCP_INIT_PAYLOAD,
                headers=headers,
            ),
            timeout=timeout,
        )
        response.raise_for_status()

        self._session_id = response.headers.get("Mcp-Session-Id")

        session_headers: dict[str, str] = dict(headers)
        if self._session_id:
            session_headers["Mcp-Session-Id"] = self._session_id

        self._wrapper = MCPClientWrapper(
            transport="http",
            url=RENDER_MCP_URL,
            headers=session_headers,
        )

    async def get_tools(self, timeout: float = 10.0) -> list[StructuredTool]:
        """Initialize the Render MCP session and return LangChain tools."""
        if self._wrapper is None:
            await self._initialize(timeout)
        assert self._wrapper is not None
        return await self._wrapper.get_tools(timeout=timeout)

    async def close(self) -> None:
        """Tear down the MCP session and the underlying HTTP client."""
        if self._wrapper is not None:
            await self._wrapper.close()
            self._wrapper = None
        await self._http.aclose()


def render_mcp_client(api_key: str | None = None) -> RenderMCPClient:
    """Return a RenderMCPClient ready to connect to https://mcp.render.com/mcp."""
    return RenderMCPClient(api_key=api_key)


# ---------------------------------------------------------------------------
# WebMCP client (Chrome web pages interaction)
# ---------------------------------------------------------------------------
class WebMCPClient:
    """MCP client for WebMCP (@jason.today/webmcp), enabling the agent to interact

    directly with active Chrome web pages and tabs via standard MCP stdio transport.
    """

    def __init__(
        self,
        command: str = "npx",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = (
            args if args is not None else ["-y", "@jason.today/webmcp@latest", "--mcp"]
        )
        self._env = env
        self._wrapper = MCPClientWrapper(
            transport="stdio",
            command=self._command,
            args=self._args,
            env=self._env,
        )

    async def connect(self, timeout: float = 10.0) -> None:
        """Connect to the WebMCP server process."""
        await self._wrapper.connect(timeout=timeout)

    async def get_tools(self, timeout: float = 10.0) -> list[StructuredTool]:
        """Fetch all browser and page interaction tools exposed by WebMCP."""
        return await self._wrapper.get_tools(timeout=timeout)

    async def close(self) -> None:
        """Tear down WebMCP session."""
        await self._wrapper.close()


def webmcp_client(
    command: str = "npx",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> WebMCPClient:
    """Return a configured WebMCPClient instance."""
    return WebMCPClient(command=command, args=args, env=env)


# ---------------------------------------------------------------------------
# Neo4j MCP Client (Direct Cypher query/write execution)
# ---------------------------------------------------------------------------
class Neo4jMCPClient:
    """MCP client for Neo4j MCP server (python -m neo4j_mcp_server).

    Allows agents to execute Cypher queries, explore schemas, and read/write
    graph data directly in Neo4j via official MCP stdio transport.
    """

    def __init__(
        self,
        command: str = "python",
        args: list[str] | None = None,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        read_only: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args if args is not None else ["-m", "neo4j_mcp_server"]
        self._env = dict(env or {})
        # Populate standard NEO4J_MCP_* configuration environment variables
        self._env.setdefault("NEO4J_MCP_URI", uri)
        self._env.setdefault("NEO4J_MCP_USERNAME", username)
        self._env.setdefault("NEO4J_MCP_PASSWORD", password)
        self._env.setdefault("NEO4J_MCP_DATABASE", database)
        self._env.setdefault(
            "NEO4J_MCP_READ_ONLY", "false" if not read_only else "true"
        )
        self._env.setdefault("NEO4J_MCP_TELEMETRY", "false")
        self._env.setdefault("NEO4J_MCP_LOG_LEVEL", "info")
        self._env.setdefault("NEO4J_MCP_LOG_FORMAT", "text")
        self._env.setdefault("NEO4J_MCP_SCHEMA_SAMPLE_SIZE", "100")

        self._wrapper = MCPClientWrapper(
            transport="stdio",
            command=self._command,
            args=self._args,
            env=self._env,
        )

    async def connect(self, timeout: float = 10.0) -> None:
        """Connect to the Neo4j MCP server process."""
        await self._wrapper.connect(timeout=timeout)

    async def get_tools(self, timeout: float = 10.0) -> list[StructuredTool]:
        """Fetch all Cypher and graph manipulation tools provided by neo4j-mcp-server."""
        return await self._wrapper.get_tools(timeout=timeout)

    async def close(self) -> None:
        """Tear down Neo4j MCP session."""
        await self._wrapper.close()


def neo4j_mcp_client(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    read_only: bool = False,
    command: str = "python",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Neo4jMCPClient:
    """Return a configured Neo4jMCPClient instance."""
    return Neo4jMCPClient(
        command=command,
        args=args,
        uri=uri,
        username=username,
        password=password,
        database=database,
        read_only=read_only,
        env=env,
    )
