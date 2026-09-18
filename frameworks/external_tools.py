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

No PAT is hard-coded. For interactive use, drive token acquisition through
the OAuth flow exposed at GET /auth/github in api_server.py.
"""

import os
from typing import Any
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
try:
    from langchain_tavily import TavilySearch as TavilySearchResults
except ImportError:
    try:
        from langchain_tavily import TavilySearchResults
    except ImportError:
        from langchain_community.tools.tavily_search import TavilySearchResults
from frameworks.mcp_client import MCPClientWrapper

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
        tavily_api_key: str | None = None,
        tavily_max_results: int = 5,
    ) -> None:
        # Resolve GitHub token: explicit arg > GITHUB_PERSONAL_ACCESS_TOKEN > GITHUB_PAT
        self._github_token: str = (
            github_token
            or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            or os.getenv("GITHUB_PAT", "")
        )
        self._github_mcp_image = github_mcp_image

        # Standard mcpServers definition for chrome-devtools
        self._mcp_servers = mcp_servers or {
            "mcpServers": {
                "chrome-devtools": {
                    "command": "npx",
                    "args": ["-y", "chrome-devtools-mcp@latest"],
                }
            }
        }
        chrome_config = self._mcp_servers.get("mcpServers", {}).get("chrome-devtools", {})
        self._chrome_command = chrome_config.get("command", chrome_command)
        self._chrome_args = chrome_config.get("args", chrome_args or ["-y", "chrome-devtools-mcp@latest"])

        # Tavily Search Tool
        tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        self._tavily_tool = TavilySearchResults(
            max_results=tavily_max_results,
            tavily_api_key=tavily_key if tavily_key else None,
        )

        # MCP Clients
        self._mcp_clients: dict[str, MCPClientWrapper] = {}

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
                headers={"Authorization": f"Bearer {self._github_token}"},
            )

        # 2. Chrome DevTools MCP (stdio)
        self._mcp_clients["chrome-devtools"] = MCPClientWrapper(
            transport="stdio",
            command=self._chrome_command,
            args=self._chrome_args,
        )

    async def connect_all(self) -> None:
        """Connects to all configured MCP clients."""
        self._init_clients()
        for client in self._mcp_clients.values():
            try:
                await client.connect()
            except Exception:
                pass

    async def close_all(self) -> None:
        """Closes all active MCP client sessions."""
        for client in self._mcp_clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._mcp_clients.clear()

    async def get_tools(self, include_mcp: bool = True, include_search: bool = True) -> list[StructuredTool]:
        """Collects all tools from MCP servers and Tavily into a single list."""
        all_tools: list[StructuredTool] = []

        if include_search and self._tavily_tool:
            all_tools.append(self._tavily_tool)

        if include_mcp:
            self._init_clients()
            for name, client in self._mcp_clients.items():
                try:
                    tools = await client.get_tools()
                    all_tools.extend(tools)
                except Exception:
                    # In development / offline mode, gracefully skip unavailable servers
                    pass

        return all_tools
