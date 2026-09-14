"""
nodes/build_subagent.py - Build subagent definition.
Implements, verifies, commits, and deploys based on PLAN.md and ARCHITECTURE.md.
Integrated with ExternalToolsManager (Tavily search, GitHub MCP, Chrome DevTools MCP)
and MCP clients (Render MCP, Draw.io MCP, and custom MCPClientWrappers).
"""

from __future__ import annotations

import os
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_ollama import ChatOllama

from frameworks.external_tools import ExternalToolsManager
from frameworks.mcp_client import (
    MCPClientWrapper,
    RenderMCPClient,
    render_mcp_client,
    DrawioMCPClient,
    drawio_mcp_client,
)
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset, create_verification_tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.harness import HarnessGuard
from langgraph.types import interrupt

class ShellCommandArgs(BaseModel):
    command: str = Field(description="The shell command string to execute (e.g. 'git status', 'git add .', 'git commit -m ...').")

def create_shell_tool(
    backend: LocalShellBackend,
    guard: HarnessGuard | None = None,
    root_dir: str = ".",
) -> StructuredTool:
    """Creates a StructuredTool wrapping LocalShellBackend.execute directly with a HITL guardrail layer."""
    harness_guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))

    def run_shell(command: str) -> str:
        cmd_stripped = command.strip()
        signature = f"Bash({cmd_stripped})"

        # Full three-gate HITL pipeline: deny → ask (interrupt) → allow
        # Failures are automatically logged to harness/failures.md by HarnessGuard.
        try:
            harness_guard.enforce(
                signature,
                interrupt_fn=interrupt,
                description=f"Build agent requests: {cmd_stripped}",
            )
        except PermissionError as exc:
            return str(exc)

        # Gate passed — execute safely via LocalShellBackend
        res = backend.execute(cmd_stripped)
        output = (res.output or "").strip()
        status = "succeeded" if res.exit_code == 0 else f"failed with exit code {res.exit_code}"
        return f"[Command {status}]\n{output}" if output else f"[Command {status}]"

    return StructuredTool.from_function(
        func=run_shell,
        name="execute_command",
        description="Executes shell commands on the host within project root via LocalShellBackend. Protected by a HITL guardrail layer (blocks harmful commands, requests approval for sensitive operations).",
        args_schema=ShellCommandArgs,
    )

BUILD_SYSTEM_PROMPT = """You are the build-agent for an autonomous coding harness.
Your task is to implement the system specified in PLAN.md and ARCHITECTURE.md.

Tool Usage & Integration:
- Terminal & Shell Execution: Use the `execute_command` tool (or built-in `execute`) to run git commands (following `harness/GITHUB.md`) and any host shell commands needed.
- Self-Verification: Call the `run_verification` tool ('all', 'lint', or 'test') to validate your changes. It runs linting and pytest tests, returning actionable stdout/stderr tracebacks. If verification fails, inspect the traceback, resolve the issue in code, and verify again.
- External Tools: Use Tavily search (tavily_search_results_json) to search documentation, libraries, and external APIs.
- MCP Tools: Use GitHub MCP tools for repository operations (branches, commits, PRs, issues) and Chrome DevTools MCP for browser inspection and UI debugging.
- Deployment Tools: Use deploy tools and Render/Vercel MCP tools to verify and trigger staging/production deployments.

Build Loop Rules:
1. Orientation: Read AGENTS.md, PLAN.md, ARCHITECTURE.md, progress.md, harness/GITHUB.md, and check git status.
2. Decompose into todos via write_todos.
3. For each todo item:
   a. Retrieve context (codebase + docs, Tavily search if needed).
   b. Write code into files (staged write).
   c. Run 'run_verification'. If it fails, examine the stderr/traceback and fix on the next attempt.
   d. Never edit or delete a test simply to make a run pass.
   e. Commit your changes and mark the todo done.
4. After completing all todos:
   a. Run an end-to-end check before declaring the feature complete.
   b. Follow harness/GITHUB.md and run git commands via execute_command / LocalShellBackend to stage, commit, and push the project into the user's GitHub repository.
"""


async def get_build_dev_tools(
    base_tools: list[Any] | None = None,
    external_tools_manager: ExternalToolsManager | None = None,
    mcp_clients: list[MCPClientWrapper] | None = None,
    include_deploy: bool = True,
    root_dir: str = ".",
) -> tuple[list[Any], ExternalToolsManager, list[MCPClientWrapper]]:
    """Gathers and resolves all tools for the build subagent:

    1. Base / deployment tools (DeployToolset)
    2. External tools (Tavily search, GitHub MCP, Chrome DevTools MCP) — each
       wrapped through HarnessGuard so permissions.json is enforced before
       any external tool call.
    3. Dedicated MCP clients (Render MCP, Draw.io MCP, custom MCPClientWrappers)

    Returns:
        (tools_list, external_tools_manager, active_mcp_clients)
    """
    tools: list[Any] = list(base_tools or [])

    # Shared guard instance for this tool-set build.
    harness_guard = HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))

    # 1. Base Deployment Tools
    if include_deploy and not any(getattr(t, "name", "") == "deploy_to_render" for t in tools):
        try:
            deploy_toolset = DeployToolset(root_dir=root_dir)
            tools.extend(deploy_toolset.get_tools())
        except Exception:
            pass

    # 2. External Tools Manager (Tavily + GitHub MCP + Chrome DevTools MCP)
    ext_mgr = external_tools_manager or ExternalToolsManager()
    try:
        ext_tools = await ext_mgr.get_tools(include_mcp=True, include_search=True)
        for t in ext_tools:
            if t not in tools:
                # Wrap every external / MCP tool through the HITL guard so that
                # patterns like "mcp__github__*" in permissions.json are enforced.
                tools.append(harness_guard.wrap_tool_with_guard(t, interrupt_fn=interrupt))
    except Exception:
        # Fallback to search tool if MCP connections are offline
        if ext_mgr._tavily_tool and ext_mgr._tavily_tool not in tools:
            tools.append(harness_guard.wrap_tool_with_guard(ext_mgr._tavily_tool, interrupt_fn=interrupt))

    # 3. Dedicated MCP clients
    clients = list(mcp_clients or [])
    if os.getenv("RENDER_API_KEY") and not any(isinstance(c, RenderMCPClient) for c in clients):
        clients.append(render_mcp_client())

    for client in clients:
        try:
            mcp_tools = await client.get_tools()
            for t in mcp_tools:
                if t not in tools:
                    tools.append(harness_guard.wrap_tool_with_guard(t, interrupt_fn=interrupt))
        except Exception:
            pass

    # 4. Programmatic Self-Verification Tool — wired with interrupt for real HITL escalation
    if not any(getattr(t, "name", "") == "run_verification" for t in tools):
        try:
            tools.append(create_verification_tool(root_dir=root_dir, interrupt_fn=interrupt))
        except Exception:
            pass

    # 5. LocalShellBackend Terminal & Shell Execution Tool (with HITL Guardrail Layer)
    if not any(getattr(t, "name", "") == "execute_command" for t in tools):
        # Hard failure here: if the guarded shell tool cannot be created the build
        # subagent must not run without it, as unguarded shell access would be unsafe.
        tools.append(create_shell_tool(LocalShellBackend(root_dir=root_dir), root_dir=root_dir))

    return tools, ext_mgr, clients


async def make_build_subagent(
    root_dir: str = ".",
    model: str = "gpt-oss:120b-cloud",
    temperature: float = 0.2,
    extra_tools: list[Any] | None = None,
    external_tools_manager: ExternalToolsManager | None = None,
    mcp_clients: list[MCPClientWrapper] | None = None,
) -> tuple[Any, ExternalToolsManager, list[MCPClientWrapper]]:
    """Build a deep build subagent wired with LocalShellBackend, ExternalToolsManager,

    and MCPClientWrapper tools.

    Usage::

        agent, ext_mgr, mcp_clients = await make_build_subagent(root_dir=".")
        try:
            await agent.ainvoke({...})
        finally:
            await ext_mgr.close_all()
            for client in mcp_clients:
                await client.close()

    Returns:
        (deep_agent, external_tools_manager, active_mcp_clients)
    """
    llm = ChatOllama(model=model, temperature=temperature)
    backend = LocalShellBackend(root_dir=root_dir)

    tools, ext_mgr, active_mcp_clients = await get_build_dev_tools(
        base_tools=extra_tools,
        external_tools_manager=external_tools_manager,
        mcp_clients=mcp_clients,
        root_dir=root_dir,
    )

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=BUILD_SYSTEM_PROMPT,
        tools=tools,
    )
    return agent, ext_mgr, active_mcp_clients


def build_dev_subagent(
    all_tools: list[Any] | None = None,
    external_tools: ExternalToolsManager | None = None,
    mcp_clients: list[MCPClientWrapper] | None = None,
    root_dir: str = ".",
    backend: LocalShellBackend | None = None,
    guard: HarnessGuard | None = None,
) -> dict[str, Any]:
    """Constructs the build subagent specification with assigned tools,

    integrating LocalShellBackend execution, ExternalToolsManager, and MCP clients.
    All external / MCP tools are wrapped through HarnessGuard so that
    permissions.json is enforced before any external tool call.
    """
    tools = list(all_tools or [])

    # Shared guard for this subagent build.
    harness_guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    shell_backend = backend or LocalShellBackend(root_dir=root_dir)

    ext_mgr = external_tools or ExternalToolsManager()
    # Immediately wire synchronous external tools (like Tavily search) — guarded.
    if ext_mgr._tavily_tool and ext_mgr._tavily_tool not in tools:
        tools.append(harness_guard.wrap_tool_with_guard(ext_mgr._tavily_tool, interrupt_fn=interrupt))

    # Immediately wire self-verification tool with real HITL escalation
    if not any(getattr(t, "name", "") == "run_verification" for t in tools):
        try:
            tools.append(create_verification_tool(root_dir=root_dir, interrupt_fn=interrupt))
        except Exception:
            pass

    # LocalShellBackend Terminal & Shell Execution Tool (with HITL Guardrail Layer)
    # Hard failure: the build subagent must not run without a guarded shell tool.
    if not any(getattr(t, "name", "") == "execute_command" for t in tools):
        tools.append(create_shell_tool(shell_backend, guard=harness_guard, root_dir=root_dir))

    return {
        "name": "build-agent",
        "description": "Use AFTER design-agent. Implements, verifies, commits, deploys with external tools and MCP.",
        "system_prompt": BUILD_SYSTEM_PROMPT,
        "tools": tools,
        "factory": make_build_subagent,
        "external_tools": ext_mgr,
        "mcp_clients": mcp_clients or [],
    }


# ---------------------------------------------------------------------------
# Convenience static spec for declarative harnesses
# ---------------------------------------------------------------------------
build_subagent: dict[str, Any] = {
    "name": "build-agent",
    "description": "Use AFTER design-agent. Implements, verifies, commits, deploys with external tools and MCP.",
    "system_prompt": BUILD_SYSTEM_PROMPT,
    "factory": make_build_subagent,
}
