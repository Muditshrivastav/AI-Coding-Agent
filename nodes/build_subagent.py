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
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from frameworks.external_tools import ExternalToolsManager
from frameworks.deepagents_backend import GuardedShellBackend
from frameworks.mcp_client import (
    MCPClientWrapper,
    RenderMCPClient,
    render_mcp_client,
    DrawioMCPClient,
    drawio_mcp_client,
)
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset, create_verification_tool
from tools.api_tools import APIToolset
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent.guardrail import HarnessGuard
from langgraph.types import interrupt

BuildMCPClient = MCPClientWrapper | RenderMCPClient


class ProbabilisticDecisionArgs(BaseModel):
    state: str = Field(
        description="The context or problem description (e.g., 'Deploy failed with status 500' or 'Lint failed on auth module')."
    )
    decision_type: str = Field(
        default="triage",
        description="Type of decision to make: 'triage' (urgency, severity, team) or 'strategy' (retry, roll back, investigate, escalate).",
    )


def create_probabilistic_decision_tool() -> StructuredTool:
    """Create a decision tool backed by the local Ollama Nimble model."""
    decision_model = ChatOllama(model="nimble:latest", temperature=0.2)

    def run_decision(state: str, decision_type: str = "triage") -> str:
        if decision_type == "strategy":
            instructions = (
                "Assess whether the issue is likely auto-fixable, recommend one action "
                "(investigate, retry, rollback, or escalate), and state your confidence."
            )
        else:
            instructions = (
                "Assess urgency, classify the issue as build, test, lint, or deploy, "
                "and state its severity as minor, blocking, or critical."
            )

        response = decision_model.invoke(
            f"{instructions}\nBe concise and base your assessment only on the context below.\n\n"
            f"Context:\n{state}"
        )
        if not isinstance(response.content, str):
            raise TypeError("Ollama Nimble returned a non-text decision response.")
        return f"[Ollama Nimble Decision]\n{response.content}"

    return StructuredTool.from_function(
        func=run_decision,
        name="probabilistic_decision",
        description=(
            "Uses the local Ollama nimble:latest model to assess build issues for triage "
            "or recommend a recovery strategy."
        ),
        args_schema=ProbabilisticDecisionArgs,
    )


class ShellCommandArgs(BaseModel):
    command: str = Field(
        alias="cmd", 
        description="The shell command string to execute (e.g. 'git status', 'git add .', 'git commit -m ...')."
    )
    timeout_seconds: int = Field(
        default=900,
        ge=1,
        le=1800,
        description="Maximum command duration in seconds (1-1800).",
    )

def create_shell_tool(
    backend: Any,
    guard: HarnessGuard | None = None,
    root_dir: str = ".",
) -> StructuredTool:
    """Creates a StructuredTool wrapping LocalShellBackend.execute directly with a HITL guardrail layer."""
    harness_guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))

    def run_shell(command: str, timeout_seconds: int = 900) -> str:
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
        execution_backend = getattr(backend, "raw_backend", backend)
        res = execution_backend.execute(cmd_stripped, timeout=timeout_seconds)
        output = (res.output or "").strip()
        status = "succeeded" if res.exit_code == 0 else f"failed with exit code {res.exit_code}"
        return f"[Command {status}]\n{output}" if output else f"[Command {status}]"

    return StructuredTool.from_function(
        func=run_shell,
        name="execute_command",
        description=(
            "Runs host shell commands from the workspace root with the user's PATH and "
            "environment, including package installation and Git commands. Use "
            "timeout_seconds up to 1800 for long installs/builds. Protected by HarnessGuard."
        ),
        args_schema=ShellCommandArgs,
    )

BUILD_SYSTEM_PROMPT = """You are the build-agent for an autonomous coding harness.
Your task is to implement the system specified in PLAN.md and ARCHITECTURE.md.

FILESYSTEM PATH RULES (CRITICAL — READ FIRST):
- ALL filesystem tools (ls, read_file, write_file, edit_file, glob, grep) require POSIX-style virtual paths starting with /
- NEVER use Windows paths like C:\\path or D:\\path — these will fail with an error
- The workspace root is mounted at / so use /agent/core.py not D:\\...\\agent\\core.py
- To list the workspace root use: ls(path="/")
- To read a file use: read_file(file_path="/nodes/build_subagent.py")
- Shell commands via execute_command use the host shell and DO accept normal paths/commands

Tool Usage & Integration:
- Permission requests: If the task requires a shell command or file mutation, invoke the corresponding tool instead of waiting for approval in chat or silently skipping the action. HarnessGuard intercepts it before execution and pauses the run for approval when permissions.json says 'ask'. Continue from the resumed tool result after approval. If denied, do not retry through another tool or method.
- Terminal & Shell Execution: `terminal_run` uses a persistent workspace shell shared with the orchestrator and retains its working directory/environment across calls. Use `terminal_read` after a timeout and `terminal_close` when finished. Use `execute_command` for isolated commands. Do not tell the user shell/Git access is unavailable instead of invoking a tool. HarnessGuard checks every command before execution; set timeouts up to 1800 seconds for long installs/builds.
- Filesystem Tools: Use `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` with POSIX virtual paths (e.g., /agent/core.py, /nodes/build_subagent.py). Always start with ls(path="/") to explore the workspace.
- Probabilistic Decisions (Low-token): Use the `probabilistic_decision` tool for triage and strategy decisions when encountering errors or needing structured probabilistic categorization without heavy reasoning token overhead.
- Self-Verification: Call the `run_verification` tool ('all', 'lint', or 'test') to validate your changes. It runs linting and pytest tests, returning actionable stdout/stderr tracebacks. If verification fails, inspect the traceback, resolve the issue in code, and verify again.
- API Tools & Web Search: Use `tavily_web_search` for web/doc search, GitHub REST tools (`github_create_issue`, `github_list_issues`, `github_create_pr`, `github_repo_info`, `github_list_commits`) for repository management, `supabase_query` for database interactions, `render_service_status` for deployment inspection, `groq_infer` for fast sub-LLM tasks, and `langsmith_recent_runs` for debugging traces.
- External Tools: Use Tavily search (tavily_search_results_json) to search documentation, libraries, and external APIs.
- MCP Tools: Use GitHub MCP tools for repository operations (branches, commits, PRs, issues) and Chrome DevTools MCP for browser inspection and UI debugging.
- Deployment Tools: Use deploy tools (`deploy_to_render`) and Render/Vercel MCP tools to verify and trigger staging/production deployments.

Build Loop Rules:
1. Orientation: Start with ls(path="/") to explore, then read /AGENTS.md, /PLAN.md, /harness/GITHUB.md, and check git status via execute_command.
2. Decompose into todos via write_todos.
3. For each todo item:
   a. Retrieve context (codebase + docs, Tavily search if needed).
   b. Write code into files (staged write).
   c. Run 'run_verification'. If it fails, examine the stderr/traceback and fix on the next attempt.
   d. Never edit or delete a test simply to make a run pass.
   e. Commit your changes and mark the todo done.
4. After completing all todos:
   a. Run an end-to-end check before declaring the feature complete.
   b. Follow /harness/GITHUB.md and run git commands via execute_command to stage, commit, and push.
"""


async def get_build_dev_tools(
    base_tools: list[Any] | None = None,
    external_tools_manager: ExternalToolsManager | None = None,
    mcp_clients: list[BuildMCPClient] | None = None,
    include_deploy: bool = True,
    root_dir: str = ".",
    guard: HarnessGuard | None = None,
) -> tuple[list[Any], ExternalToolsManager, list[BuildMCPClient]]:
    """Gathers and resolves all tools for the build subagent:

    1. Base / deployment tools (DeployToolset)
    2. External tools (Tavily search, GitHub MCP, Chrome DevTools MCP) — each
       wrapped through HarnessGuard so permissions.json is enforced before
       any external tool call.
    3. Dedicated MCP clients (Render MCP, Draw.io MCP, custom MCPClientWrappers)

    Returns:
        (tools_list, external_tools_manager, active_mcp_clients)
    """
    # Shared guard instance for this tool-set build.
    harness_guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    tools: list[Any] = [
        harness_guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt)
        for tool in (base_tools or [])
    ]

    # 1. Base Deployment & API Tools
    if include_deploy and not any(getattr(t, "name", "") == "deploy_to_render" for t in tools):
        try:
            deploy_toolset = DeployToolset(root_dir=root_dir)
            tools.extend(
                harness_guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt)
                for tool in deploy_toolset.get_tools()
            )
        except Exception:
            pass

    try:
        api_toolset = APIToolset()
        for api_tool in api_toolset.get_tools():
            if not any(getattr(t, "name", "") == api_tool.name for t in tools):
                tools.append(harness_guard.wrap_tool_with_guard(api_tool, interrupt_fn=interrupt))
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
    clients: list[BuildMCPClient] = list(mcp_clients or [])
    if os.getenv("RENDER_API_KEY") and not any(isinstance(c, RenderMCPClient) for c in clients):
        clients.append(render_mcp_client())

    for client in clients:
        try:
            import asyncio
            mcp_tools = await asyncio.wait_for(client.get_tools(), timeout=5.0)
            for t in mcp_tools:
                if t not in tools:
                    tools.append(harness_guard.wrap_tool_with_guard(t, interrupt_fn=interrupt))
        except Exception:
            pass

    # 4. Programmatic Self-Verification Tool — wired with interrupt for real HITL escalation
    if not any(getattr(t, "name", "") == "run_verification" for t in tools):
        try:
            verification_tool = create_verification_tool(root_dir=root_dir, interrupt_fn=interrupt)
            tools.append(harness_guard.wrap_tool_with_guard(verification_tool, interrupt_fn=interrupt))
        except Exception:
            pass

    # 5. Ollama Nimble probabilistic decision tool.
    if not any(getattr(t, "name", "") == "probabilistic_decision" for t in tools):
        tools.append(
            harness_guard.wrap_tool_with_guard(
                create_probabilistic_decision_tool(),
                interrupt_fn=interrupt,
            )
        )

    # 6. LocalShellBackend Terminal & Shell Execution Tool (with HITL Guardrail Layer)
    if not any(getattr(t, "name", "") == "execute_command" for t in tools):
        # Hard failure here: if the guarded shell tool cannot be created the build
        # subagent must not run without it, as unguarded shell access would be unsafe.
        tools.append(create_shell_tool(
            LocalShellBackend(root_dir=root_dir, timeout=900, inherit_env=True),
            guard=harness_guard,
            root_dir=root_dir,
        ))

    return tools, ext_mgr, clients



async def make_build_subagent(
    root_dir: str = ".",
    model: str = "qwen/qwen3.8-27b",
    temperature: float = 0.2,
    extra_tools: list[Any] | None = None,
    external_tools_manager: ExternalToolsManager | None = None,
    mcp_clients: list[BuildMCPClient] | None = None,
) -> tuple[Any, ExternalToolsManager, list[BuildMCPClient]]:
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
    primary_llm = ChatGroq(model=model, temperature=temperature)
    fallback_llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=temperature, num_ctx=8192)
    llm: Any = primary_llm.with_fallbacks([fallback_llm])
    guard = HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    raw_backend = LocalShellBackend(root_dir=root_dir, timeout=900, inherit_env=True)
    backend = GuardedShellBackend(raw_backend, guard)

    tools, ext_mgr, active_mcp_clients = await get_build_dev_tools(
        base_tools=extra_tools,
        external_tools_manager=external_tools_manager,
        mcp_clients=mcp_clients,
        root_dir=root_dir,
        guard=guard,
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
    mcp_clients: list[BuildMCPClient] | None = None,
    root_dir: str = ".",
    backend: Any | None = None,
    guard: HarnessGuard | None = None,
) -> dict[str, Any]:
    """Constructs the build subagent specification with assigned tools,

    integrating LocalShellBackend execution, ExternalToolsManager, and MCP clients.
    All external / MCP tools are wrapped through HarnessGuard so that
    permissions.json is enforced before any external tool call.
    """
    # Shared guard for this subagent build.
    harness_guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    tools = [
        harness_guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt)
        for tool in (all_tools or [])
    ]
    shell_backend = backend or GuardedShellBackend(
        LocalShellBackend(root_dir=root_dir, timeout=900, inherit_env=True),
        harness_guard,
    )

    ext_mgr = external_tools or ExternalToolsManager()
    # Immediately wire synchronous external tools (like Tavily search) — guarded.
    if ext_mgr._tavily_tool and ext_mgr._tavily_tool not in tools:
        tools.append(harness_guard.wrap_tool_with_guard(ext_mgr._tavily_tool, interrupt_fn=interrupt))

    # Immediately wire API Tools (Tavily, GitHub REST, Supabase, Render, Groq, LangSmith) — guarded
    try:
        api_toolset = APIToolset()
        for api_tool in api_toolset.get_tools():
            if not any(getattr(t, "name", "") == api_tool.name for t in tools):
                tools.append(harness_guard.wrap_tool_with_guard(api_tool, interrupt_fn=interrupt))
    except Exception:
        pass

    # Immediately wire self-verification tool with real HITL escalation
    if not any(getattr(t, "name", "") == "run_verification" for t in tools):
        try:
            tools.append(create_verification_tool(root_dir=root_dir, interrupt_fn=interrupt))
        except Exception:
            pass

    # Immediately wire the Ollama Nimble probabilistic decision tool.
    if not any(getattr(t, "name", "") == "probabilistic_decision" for t in tools):
        tools.append(create_probabilistic_decision_tool())

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
