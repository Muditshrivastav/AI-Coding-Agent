"""
frameworks/external_tools.py - Unified external tool manager.
Combines external MCP servers (GitHub via MCP SDK, Chrome DevTools) and
Tavily search into a single unified class.

GitHub token sourcing (in priority order)
-----------------------------------------
1.  Explicit ``github_token`` argument — pass ``vault.get(user_id)`` from
    ``tools.github_oauth`` after the user completes the OAuth flow.
2.  ``GITHUB_PERSONAL_ACCESS_TOKEN`` env var (legacy / CI fallback).
3.  ``GITHUB_PAT`` env var (older legacy fallback).
4.  ``GITHUB_TOKEN`` env var.

No PAT is hard-coded. For interactive use, drive token acquisition through
the OAuth flow exposed at GET /auth/github in api_server.py.
"""

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch as TavilySearchResults

from frameworks.mcp_client import MCPClientWrapper

logger = logging.getLogger(__name__)

load_dotenv()

# Docker image kept as a constant for optional manual use; not auto-launched.
_GITHUB_MCP_IMAGE = "ghcr.io/github/github-mcp-server"


class ExternalToolsManager:
    """Consolidated manager for all external MCP servers and Tavily search.

    GitHub MCP is launched via Docker (stdio transport) using the official
    ``ghcr.io/github/github-mcp-server`` image so no HTTP session handshake
    is required.

    Manages connections, tool creation, and lifecycle in one place.
    """

    def __init__(
        self,
        github_token: str | None = None,
        github_mcp_image: str = _GITHUB_MCP_IMAGE,
        mcp_servers: dict[str, Any] | None = None,
        chrome_command: str = "npx",
        chrome_args: list[str] | None = None,
        webmcp_command: str = "npx",
        webmcp_args: list[str] | None = None,
        tavily_api_key: str | None = None,
        tavily_max_results: int = 5,
    ) -> None:
        # Resolve GitHub token: explicit arg > GITHUB_PERSONAL_ACCESS_TOKEN > GITHUB_PAT
        self._github_token: str = (
            github_token
            or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            or os.getenv("GITHUB_PAT", "")
            or os.getenv("GITHUB_TOKEN", "")
        )
        self._github_mcp_image = github_mcp_image

        # Standard mcpServers definition for chrome-devtools and webmcp
        self._mcp_servers = mcp_servers or {
            "mcpServers": {
                "chrome-devtools": {
                    "command": "npx",
                    "args": ["-y", "chrome-devtools-mcp@latest"],
                },
                "webmcp": {
                    "command": "npx",
                    "args": ["-y", "@jason.today/webmcp@latest", "--mcp"],
                },
            }
        }
        servers = self._mcp_servers.get("mcpServers", {})
        chrome_config = servers.get("chrome-devtools", {})
        self._chrome_command = chrome_config.get("command", chrome_command)
        self._chrome_args = chrome_config.get(
            "args", chrome_args or ["-y", "chrome-devtools-mcp@latest"]
        )

        webmcp_config = servers.get("webmcp", {})
        self._webmcp_command = webmcp_config.get("command", webmcp_command)
        self._webmcp_args = webmcp_config.get(
            "args", webmcp_args or ["-y", "@jason.today/webmcp@latest", "--mcp"]
        )

        # Tavily Search Tool
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        self._tavily_tool: BaseTool | None = TavilySearchResults(
            max_results=tavily_max_results,
            tavily_api_key=tavily_key if tavily_key else None,
        )

        # MCP Clients
        self._mcp_clients: dict[str, MCPClientWrapper] = {}
        self._server_status: dict[str, str] = {}
        self._loaded_tool_names: set[str] = set()

    @property
    def loaded_tool_names(self) -> list[str]:
        return sorted(self._loaded_tool_names)

    @property
    def server_status(self) -> dict[str, str]:
        return dict(self._server_status)

    @property
    def mcp_clients(self) -> list[MCPClientWrapper]:
        return list(self._mcp_clients.values())

    def _init_clients(self) -> None:
        if self._mcp_clients:
            return

        # 1. GitHub MCP via streamable HTTP (token sourced from OAuth vault at runtime)
        #    Only wired when a token is available — avoids connecting an unauthenticated client.
        #    Official remote MCP endpoint: https://api.githubcopilot.com/mcp/
        if self._github_token:
            self._mcp_clients["github"] = MCPClientWrapper(
                transport="http",
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": "Bearer " + self._github_token},
            )

        else:
            self._server_status["github"] = "skipped: no GitHub token configured"

        # 2. Chrome DevTools MCP (stdio)
        self._mcp_clients["chrome-devtools"] = MCPClientWrapper(
            transport="stdio",
            command=self._chrome_command,
            args=self._chrome_args,
        )

        # 3. WebMCP (stdio) for Chrome web page / tab interaction
        self._mcp_clients["webmcp"] = MCPClientWrapper(
            transport="stdio",
            command=self._webmcp_command,
            args=self._webmcp_args,
        )

    async def connect_all(self) -> None:
        """Connect to every configured server with isolated startup failures."""
        self._init_clients()
        timeout = float(os.getenv("MCP_SERVER_TIMEOUT_SECONDS", "10"))
        for name, client in self._mcp_clients.items():
            try:
                await asyncio.wait_for(client.connect(), timeout=timeout)
                self._server_status[name] = "connected"
            except Exception as exc:
                self._server_status[name] = f"unavailable: {exc}"
                logger.warning("MCP server %s failed to connect: %s", name, exc)

    async def close_all(self) -> None:
        """Closes all active MCP client sessions."""
        for name, client in self._mcp_clients.items():
            try:
                await client.close()
            except Exception as exc:
                logger.warning("Failed to close MCP server %s: %s", name, exc)
        self._mcp_clients.clear()

    async def get_tools(
        self, include_mcp: bool = True, include_search: bool = True
    ) -> list[BaseTool]:
        """Collects all tools from MCP servers and Tavily into a single list."""
        all_tools: list[BaseTool] = []

        if include_search and self._tavily_tool:
            all_tools.append(self._tavily_tool)
            self._loaded_tool_names.add(self._tavily_tool.name)

        if include_mcp:
            self._init_clients()
            timeout = float(os.getenv("MCP_SERVER_TIMEOUT_SECONDS", "10"))
            for name, client in self._mcp_clients.items():
                try:
                    tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
                    all_tools.extend(tools)
                    self._loaded_tool_names.update(tool.name for tool in tools)
                    self._server_status[name] = f"loaded {len(tools)} tools"
                except Exception as exc:
                    self._server_status[name] = f"unavailable: {exc}"
                    logger.warning("MCP server %s is unavailable: %s", name, exc)

        return all_tools
