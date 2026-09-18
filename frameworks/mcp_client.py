"""
frameworks/mcp_client.py - Official MCP SDK client wrapper.
Wraps one MCP server (stdio or HTTP) and converts its tools into LangChain StructuredTools.
"""

import os
from contextlib import AsyncExitStack
from typing import Any

import httpx
from pydantic import create_model
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.sse import sse_client as streamablehttp_client
except ImportError:
    try:
        from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    except ImportError:
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            streamablehttp_client = None  # type: ignore[assignment]
from langchain_core.tools import StructuredTool


class MCPClientWrapper:
    """Wraps one MCP server (stdio or streamable HTTP) via the official MCP Python SDK

    and converts exposed tools into LangChain StructuredTools.
    """

    def __init__(self, transport: str, **kwargs: Any) -> None:
        self._transport = transport.lower()  # "stdio" | "http"
        self._kwargs = kwargs
        self._session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self) -> None:
        if self._session is not None:
            return

        if self._transport == "stdio":
            params = StdioServerParameters(
                command=self._kwargs["command"],
                args=self._kwargs.get("args", []),
                env=self._kwargs.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif self._transport == "http":
            if streamablehttp_client is None:
                raise ImportError("Neither 'mcp.client.sse.sse_client' nor 'streamable_http_client' is available in your installed MCP SDK.")
            client_kwargs: dict[str, Any] = {"url": self._kwargs["url"]}
            if self._kwargs.get("headers"):
                client_kwargs["headers"] = self._kwargs["headers"]
            conn = await self._stack.enter_async_context(
                streamablehttp_client(**client_kwargs)  # type: ignore[operator]
            )
            # Both sse_client and streamable_http_client return (read, write) or (read, write, _)
            if isinstance(conn, (tuple, list)):
                read, write = conn[0], conn[1]
            else:
                read, write = conn.read, conn.write
        else:
            raise ValueError(f"Unsupported transport: {self._transport}. Must be 'stdio' or 'http'.")

        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None

    def _make_tool(self, mcp_tool: Any) -> StructuredTool:
        schema = getattr(mcp_tool, "inputSchema", None) or {"type": "object", "properties": {}}
        fields = {name: (str, ...) for name in schema.get("properties", {})}
        args_model = create_model(f"{mcp_tool.name}_Args", **fields) if fields else None

        async def _call(**kwargs: Any) -> str:
            if self._session is None:
                raise RuntimeError("MCPClientWrapper is not connected. Call connect() first.")
            result = await self._session.call_tool(mcp_tool.name, arguments=kwargs)
            return "\n".join(b.text for b in result.content if hasattr(b, "text"))

        return StructuredTool.from_function(
            coroutine=_call,
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            args_schema=args_model,
        )

    async def get_tools(self) -> list[StructuredTool]:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        result = await self._session.list_tools()
        return [self._make_tool(t) for t in result.tools]


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

    async def _initialize(self) -> None:
        """POST the MCP initialize request and capture the Mcp-Session-Id header."""
        response = await self._http.post(
            DRAWIO_MCP_URL,
            json=_MCP_INIT_PAYLOAD,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
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

    async def get_tools(self) -> list[StructuredTool]:
        """Initialize the draw.io session (if needed) and return LangChain tools."""
        if self._wrapper is None:
            await self._initialize()
        assert self._wrapper is not None
        return await self._wrapper.get_tools()

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

    async def _initialize(self) -> None:
        """POST the MCP initialize request with Bearer authorization and capture session ID."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = await self._http.post(
            RENDER_MCP_URL,
            json=_MCP_INIT_PAYLOAD,
            headers=headers,
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

    async def get_tools(self) -> list[StructuredTool]:
        """Initialize the Render MCP session and return LangChain tools."""
        if self._wrapper is None:
            await self._initialize()
        assert self._wrapper is not None
        return await self._wrapper.get_tools()

    async def close(self) -> None:
        """Tear down the MCP session and the underlying HTTP client."""
        if self._wrapper is not None:
            await self._wrapper.close()
            self._wrapper = None
        await self._http.aclose()


def render_mcp_client(api_key: str | None = None) -> RenderMCPClient:
    """Return a RenderMCPClient ready to connect to https://mcp.render.com/mcp."""
    return RenderMCPClient(api_key=api_key)

