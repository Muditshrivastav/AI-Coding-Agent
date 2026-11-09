"""
agent/core.py - CodingAgentHarness: single entry point for running and resuming agent executions.

Also exposes ``build_harness_graph()`` which wires every agent component into a single
LangGraph StateGraph:

    input_guard → planning_node → design_node → graphrag_node → build_node
                → deploy_node → verify_node → [api_node | repair_node]
                → session_node → output_node

The graph can be used directly or the harness ``run()`` method will fall back to it
when ``use_pipeline_graph=True`` is passed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any, Required, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Annotated, Command, interrupt

from agent.guardrail import HarnessGuard
from agent.repair_loop import build_repair_graph
from agent.session_manager import SessionManager
from agent.state import AgentState
from frameworks.agents_md_writer import generate_agents_md
from frameworks.deepagents_backend import DeepAgentsBackend
from frameworks.evaluation import HarnessEvaluator, LangSmithTracer
from frameworks.external_tools import ExternalToolsManager
from frameworks.langgraph_runtime import LangGraphRuntime
from frameworks.model_config import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
)
from graphrag.ingest_tool import create_repo_ingest_tool
from graphrag.tool import create_graphrag_retriever_tool
from nodes.build_subagent import build_dev_subagent, get_build_dev_tools
from nodes.design_subagent import design_subagent, make_design_subagent
from nodes.planning_subagent import make_planning_subagent, planning_subagent
from skills.registry import skill_library
from tools.api_tools import APIToolset
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset

logger = logging.getLogger(__name__)


# ===========================================================================
# LangGraph Pipeline Graph State
# ===========================================================================

class HarnessGraphState(TypedDict, total=False):
    """Unified state that flows through the full harness pipeline graph."""

    # ── Core request fields ────────────────────────────────────────────────
    user_request: Required[str]
    conversation_history: str
    execution_thread_id: Required[str]
    model_id: str
    thread_id: str

    # ── Stage tracking ─────────────────────────────────────────────────────
    graph_stage: str               # Current active node name
    start_approved: bool           # Whether input_guard passed

    # ── Subagent outputs ───────────────────────────────────────────────────
    plan_output: str               # PLAN.md content from planning subagent
    arch_output: str               # ARCHITECTURE.md content from design subagent
    build_output: str              # Final build subagent output

    # ── Tool subgraph state ────────────────────────────────────────────────
    search_results: list[dict[str, Any]]     # grep/glob results
    files_read: list[dict[str, Any]]         # read_file results
    files_written: list[dict[str, Any]]      # write_file results
    files_edited: list[dict[str, Any]]       # edit_file results
    test_result: dict[str, Any]              # pytest run result
    git_result: dict[str, Any]               # git push result
    tool_subgraph_status: str                # "pending" | "passed" | "failed"

    # ── GraphRAG ───────────────────────────────────────────────────────────
    context_docs: list[dict[str, Any]]       # Retrieved context from GraphRAG

    # ── Deploy / verify / API ──────────────────────────────────────────────
    deploy_result: dict[str, Any]
    verify_result: dict[str, Any]
    api_result: dict[str, Any]

    # ── Repair loop ────────────────────────────────────────────────────────
    repair_round: int
    max_repair_rounds: int
    repair_feedback: str
    failures: list[str]

    # ── Output ─────────────────────────────────────────────────────────────
    status: str
    final_output: str
    messages: Annotated[list[Any], add_messages]


# ===========================================================================
# Tool Subgraph — grep/glob → read → write → edit → pytest → git push
# ===========================================================================

def build_tool_subgraph(
    backend: Any,
    guard: Any,
    root_dir: str = ".",
) -> Any:
    """Build a LangGraph subgraph for the filesystem-tool pipeline.

    Pipeline flow::

        search_node  (grep + glob)
            │
            ▼
        read_node    (read_file for matched files)
            │
            ▼
        write_node   (write_file for new files)
            │
            ▼
        edit_node    (edit_file for existing files)
            │
            ▼
        test_node    (run pytest via shell)
            │
            ├─ passed  → push_node (git add/commit/push)
            └─ failed  → END (caller handles repair)

    Args:
        backend: A ``GuardedShellBackend`` for file and shell operations.
        guard: ``HarnessGuard`` for permission enforcement.
        root_dir: Workspace root directory.

    Returns:
        A compiled LangGraph ``StateGraph`` usable as a subgraph node.
    """

    class ToolSubgraphState(TypedDict, total=False):
        """Internal state for the tool subgraph."""
        search_results: list[dict[str, Any]]
        files_read: list[dict[str, Any]]
        files_written: list[dict[str, Any]]
        files_edited: list[dict[str, Any]]
        test_result: dict[str, Any]
        git_result: dict[str, Any]
        tool_subgraph_status: str
        user_request: str
        build_output: str

    # ── search_node: grep + glob ───────────────────────────────────────
    async def search_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Run grep and glob to discover relevant files."""
        results: list[dict[str, Any]] = []
        request = state.get("user_request", "")
        build_output = state.get("build_output", "")

        # Extract file patterns from build output or user request
        search_context = build_output or request
        try:
            # Glob for Python files in the workspace
            glob_result = backend.glob("**/*.py", path="/")
            if hasattr(glob_result, "matches") and glob_result.matches:
                for match in glob_result.matches[:50]:  # Cap at 50 matches
                    file_path = getattr(match, "path", str(match))
                    results.append({
                        "type": "glob",
                        "path": file_path,
                        "source": "glob(**/*.py)",
                    })
        except Exception as exc:
            logger.warning("Glob search failed: %s", exc)

        try:
            # Grep for TODO/FIXME markers or key terms from the request
            grep_terms = ["TODO", "FIXME", "HACK"]
            for term in grep_terms:
                grep_result = backend.grep(term, path="/", glob="**/*.py", max_count=20)
                if hasattr(grep_result, "matches") and grep_result.matches:
                    for match in grep_result.matches:
                        results.append({
                            "type": "grep",
                            "path": getattr(match, "file", ""),
                            "line": getattr(match, "line_number", 0),
                            "content": getattr(match, "line", ""),
                            "term": term,
                        })
        except Exception as exc:
            logger.warning("Grep search failed: %s", exc)

        return {
            "search_results": results,
            "tool_subgraph_status": "searching",
        }

    # ── read_node: read discovered files ───────────────────────────────
    async def read_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Read contents of files discovered by search."""
        files_read: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for result in state.get("search_results", []):
            path = result.get("path", "")
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            if len(files_read) >= 20:  # Don't read too many files at once
                break
            try:
                content = backend.read(path, offset=0, limit=5000)
                content_text = (
                    getattr(content, "content", str(content))
                    if not isinstance(content, str)
                    else content
                )
                files_read.append({
                    "path": path,
                    "content": content_text[:5000],
                    "size": len(content_text),
                })
            except Exception as exc:
                logger.debug("Could not read %s: %s", path, exc)

        return {"files_read": files_read}

    # ── write_node: write new files ────────────────────────────────────
    async def write_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Write new files based on build output instructions.

        The build subagent's output drives which files need creation.
        This node is a pass-through when no new files are requested.
        """
        files_written: list[dict[str, Any]] = []
        build_output = state.get("build_output", "")

        # If the build subagent explicitly produced file-write instructions
        # they would be parsed here. For now, this node records what was
        # already written by the build subagent and passes control forward.
        if not build_output:
            return {"files_written": files_written}

        return {
            "files_written": files_written,
            "tool_subgraph_status": "files_written",
        }

    # ── edit_node: edit existing files ─────────────────────────────────
    async def edit_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Apply edits to existing files.

        In the normal flow, the build subagent handles edits directly
        through the filesystem backend. This node is a checkpoint that
        records what was edited and ensures the state is consistent.
        """
        files_edited: list[dict[str, Any]] = []

        return {
            "files_edited": files_edited,
            "tool_subgraph_status": "files_edited",
        }

    # ── test_node: run pytest ──────────────────────────────────────────
    async def test_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Execute pytest in the workspace shell."""
        import asyncio as _asyncio

        try:
            result = await _asyncio.to_thread(
                backend.execute,
                "python -m pytest tests/ -v --tb=short 2>&1 || true",
            )
            output = str(getattr(result, "output", result) if not isinstance(result, str) else result)
            exit_code = getattr(result, "exit_code", 0) if not isinstance(result, str) else 0
            passed = exit_code == 0 and "failed" not in output.lower()
        except Exception as exc:
            output = f"pytest execution error: {exc}"
            passed = False
            exit_code = 1

        return {
            "test_result": {
                "passed": passed,
                "exit_code": exit_code,
                "output": output[-4000:],  # Tail to avoid state bloat
            },
            "tool_subgraph_status": "tested",
        }

    # ── push_node: git add/commit/push ─────────────────────────────────
    async def push_node(state: ToolSubgraphState) -> dict[str, Any]:
        """Stage, commit, and push changes to the remote repository."""
        import asyncio as _asyncio

        steps = [
            "git add -A",
            'git commit -m "auto: apply agent changes" --allow-empty',
            "git push origin HEAD",
        ]
        outputs: list[str] = []
        final_success = True

        for cmd in steps:
            try:
                result = await _asyncio.to_thread(backend.execute, cmd)
                output = str(
                    getattr(result, "output", result)
                    if not isinstance(result, str)
                    else result
                )
                outputs.append(f"$ {cmd}\n{output}")
            except Exception as exc:
                outputs.append(f"$ {cmd}\nERROR: {exc}")
                final_success = False
                break

        return {
            "git_result": {
                "success": final_success,
                "output": "\n".join(outputs)[-4000:],
            },
            "tool_subgraph_status": "pushed" if final_success else "push_failed",
        }

    # ── Routing ────────────────────────────────────────────────────────
    def route_after_test(state: ToolSubgraphState) -> str:
        """Route to push if tests passed, otherwise end (caller repairs)."""
        test_result = state.get("test_result", {})
        if test_result.get("passed", False):
            return "push_node"
        return END

    # ── Compile subgraph ───────────────────────────────────────────────
    graph = StateGraph(ToolSubgraphState)
    graph.add_node("search_node", search_node)
    graph.add_node("read_node", read_node)
    graph.add_node("write_node", write_node)
    graph.add_node("edit_node", edit_node)
    graph.add_node("test_node", test_node)
    graph.add_node("push_node", push_node)

    graph.add_edge(START, "search_node")
    graph.add_edge("search_node", "read_node")
    graph.add_edge("read_node", "write_node")
    graph.add_edge("write_node", "edit_node")
    graph.add_edge("edit_node", "test_node")
    graph.add_conditional_edges("test_node", route_after_test)
    graph.add_edge("push_node", END)

    return graph.compile()


# ===========================================================================
# Main Harness Pipeline Graph
# ===========================================================================

def build_harness_graph(
    *,
    planner_agent: Any,
    design_agent: Any,
    build_agent: Any,
    backend: Any,
    guard: Any,
    deploy_tools: Any,
    verification_tools: Any,
    api_tools: Any,
    graphrag_pipeline: Any | None = None,
    root_dir: str = ".",
    max_repair_rounds: int = 3,
    checkpointer: Any = None,
) -> Any:
    """Wire every agent component into a single connected LangGraph StateGraph.

    Node tiers (in execution order):

    **Subagent nodes** — orchestrate the three specialised deep-agents:
        ``input_guard`` → ``planning_node`` → ``design_node`` → ``build_node``

    **Tool nodes** — run filesystem tools, tests, and deployment:
        ``graphrag_node`` → ``tool_pipeline`` (subgraph) → ``deploy_node``
        → ``verify_node`` → ``api_node``

    **Pipeline nodes** — repair loop, session persistence, and final output:
        ``repair_node`` → (loops back to ``build_node``)
        ``session_node`` → ``output_node``

    Args:
        planner_agent: Compiled planning subagent runnable.
        design_agent: Compiled design subagent runnable.
        build_agent: Compiled build subagent runnable (dict with 'runnable').
        backend: ``GuardedShellBackend`` for filesystem/shell operations.
        guard: ``HarnessGuard`` for permission enforcement.
        deploy_tools: ``DeployToolset`` instance.
        verification_tools: ``VerificationToolset`` instance.
        api_tools: ``APIToolset`` instance.
        graphrag_pipeline: Optional ``GraphRAGPipeline`` for code retrieval.
        root_dir: Workspace root directory.
        max_repair_rounds: Max build→verify→repair cycles before stopping.
        checkpointer: LangGraph checkpointer for state persistence.

    Returns:
        A compiled ``StateGraph`` that can be invoked with ``ainvoke()``.
    """
    tool_subgraph = build_tool_subgraph(backend, guard, root_dir)

    # ── 1. input_guard: injection check + session bookkeeping ──────────
    async def input_guard(state: HarnessGraphState) -> dict[str, Any]:
        """Prompt-injection pre-flight and approval gate."""
        user_request = state["user_request"]

        # Check injection patterns
        prose = re.sub(r"```.*?```|~~~.*?~~~", "", user_request, flags=re.DOTALL)
        prose = re.sub(r"`[^`]*`", "", prose)
        for label, pattern in _INJECTION_PATTERNS:
            if pattern.search(prose):
                guard.log_failure(
                    stage="input",
                    category="prompt-injection",
                    description=f"Matched pattern '{label}' in: {user_request[:120]}",
                    resolution="Request blocked before reaching agent.",
                )
                return {
                    "start_approved": False,
                    "status": "blocked",
                    "final_output": (
                        f"Input flagged as potential prompt injection "
                        f"(matched pattern: '{label}'). Request not forwarded to agent."
                    ),
                    "graph_stage": "input_guard",
                }

        # Approval gate
        if guard is not None:
            try:
                guard.enforce(
                    "AgentRun(start)",
                    interrupt_fn=interrupt,
                    description="Start the coding agent pipeline",
                )
            except PermissionError as exc:
                return {
                    "start_approved": False,
                    "status": "blocked",
                    "final_output": str(exc),
                    "graph_stage": "input_guard",
                }

        return {
            "start_approved": True,
            "graph_stage": "input_guard",
            "repair_round": 0,
            "max_repair_rounds": state.get("max_repair_rounds", max_repair_rounds),
        }

    # ── 2. planning_node ───────────────────────────────────────────────
    async def planning_node(
        state: HarnessGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Invoke the planning subagent to produce PLAN.md and tasks.json."""
        prompt = (
            "Plan the current request. Read /AGENTS.md and /harness/TASK.md. "
            "Write PLAN.md, then call write_task_plan with a complete ordered list "
            "of independent tasks.\n\n"
            f"Request:\n{state['user_request']}"
        )
        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": f"{state['execution_thread_id']}:planning",
            },
        }
        result = await planner_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
        )
        output = ""
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                output = content.strip()
                break

        return {
            "plan_output": output or "Planning complete.",
            "graph_stage": "planning_node",
        }

    # ── 3. design_node ─────────────────────────────────────────────────
    async def design_node(
        state: HarnessGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Invoke the design subagent to produce ARCHITECTURE.md."""
        prompt = (
            "Read /PLAN.md and produce ARCHITECTURE.md. Outline components, "
            "data flow, inter-module communication, and file boundaries.\n\n"
            f"Original request:\n{state['user_request']}"
        )
        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": f"{state['execution_thread_id']}:design",
            },
        }
        result = await design_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
        )
        output = ""
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                output = content.strip()
                break

        return {
            "arch_output": output or "Architecture design complete.",
            "graph_stage": "design_node",
        }

    # ── 4. graphrag_node ───────────────────────────────────────────────
    async def graphrag_node(state: HarnessGraphState) -> dict[str, Any]:
        """Ingest the repository and retrieve relevant context."""
        context_docs: list[dict[str, Any]] = []

        if graphrag_pipeline is not None:
            try:
                # Ingest current workspace state
                ingest_tool = create_repo_ingest_tool(pipeline=graphrag_pipeline)
                await asyncio.to_thread(ingest_tool.invoke, {"path": root_dir})

                # Retrieve context for the user's request
                retriever_tool = create_graphrag_retriever_tool(
                    pipeline=graphrag_pipeline
                )
                query = state["user_request"][:500]
                raw = await asyncio.to_thread(retriever_tool.invoke, {"query": query})
                if isinstance(raw, list):
                    context_docs = raw
                elif isinstance(raw, str):
                    context_docs = [{"source": "graphrag", "content": raw[:3000]}]
            except Exception as exc:
                logger.warning("GraphRAG retrieval failed: %s", exc)

        return {
            "context_docs": context_docs,
            "graph_stage": "graphrag_node",
        }

    # ── 5. build_node ──────────────────────────────────────────────────
    async def build_node(
        state: HarnessGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Invoke the build subagent to implement code changes."""
        build_runnable = (
            build_agent.get("runnable", build_agent)
            if isinstance(build_agent, dict)
            else build_agent
        )

        context_str = ""
        for doc in state.get("context_docs", []):
            context_str += f"\n---\n{doc.get('content', '')[:1000]}"

        repair_round = state.get("repair_round", 0)
        repair_feedback = state.get("repair_feedback", "")

        if repair_round > 0 and repair_feedback:
            prompt = (
                "Continue working in the same workspace. Fix the issues from the "
                "previous round and re-run verification.\n\n"
                f"Repair feedback:\n{repair_feedback}\n\n"
                f"Original request:\n{state['user_request']}"
            )
        else:
            prompt = (
                "Implement the changes described in /PLAN.md and /ARCHITECTURE.md. "
                "Read them first, then make the smallest complete change. "
                "Run verification after editing.\n\n"
                f"Request:\n{state['user_request']}"
            )
            if context_str:
                prompt += f"\n\nRelevant context from codebase:\n{context_str[:3000]}"

        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": (
                    f"{state['execution_thread_id']}:build:round:{repair_round}"
                ),
            },
        }
        result = await build_runnable.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
        )
        output = ""
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                output = content.strip()
                break

        return {
            "build_output": output or "Build step complete.",
            "graph_stage": "build_node",
        }

    # ── 6. tool_pipeline: filesystem tool subgraph ─────────────────────
    async def tool_pipeline(state: HarnessGraphState) -> dict[str, Any]:
        """Run the tool subgraph: grep/glob → read → write → edit → pytest → git push."""
        sub_state = {
            "user_request": state.get("user_request", ""),
            "build_output": state.get("build_output", ""),
            "search_results": [],
            "files_read": [],
            "files_written": [],
            "files_edited": [],
            "test_result": {},
            "git_result": {},
            "tool_subgraph_status": "pending",
        }
        result = await tool_subgraph.ainvoke(sub_state)

        return {
            "search_results": result.get("search_results", []),
            "files_read": result.get("files_read", []),
            "files_written": result.get("files_written", []),
            "files_edited": result.get("files_edited", []),
            "test_result": result.get("test_result", {}),
            "git_result": result.get("git_result", {}),
            "tool_subgraph_status": result.get("tool_subgraph_status", "complete"),
            "graph_stage": "tool_pipeline",
        }

    # ── 7. deploy_node ─────────────────────────────────────────────────
    async def deploy_node(state: HarnessGraphState) -> dict[str, Any]:
        """Execute deployment tools (Render / Vercel)."""
        deploy_result: dict[str, Any] = {"status": "skipped"}

        # Only deploy if tests passed
        test_result = state.get("test_result", {})
        if not test_result.get("passed", False):
            deploy_result["status"] = "skipped_tests_failed"
            return {"deploy_result": deploy_result, "graph_stage": "deploy_node"}

        try:
            tools = deploy_tools.get_tools()
            if tools:
                # Use the first available deploy tool
                deploy_result = {
                    "status": "available",
                    "tools": [getattr(t, "name", str(t)) for t in tools],
                }
        except Exception as exc:
            deploy_result = {"status": "error", "error": str(exc)}

        return {"deploy_result": deploy_result, "graph_stage": "deploy_node"}

    # ── 8. verify_node ─────────────────────────────────────────────────
    async def verify_node(state: HarnessGraphState) -> dict[str, Any]:
        """Run the full verification toolset (lint + tests + static checks)."""
        verify_result: dict[str, Any] = {"passed": False}

        try:
            tools = verification_tools.get_tools()
            if tools:
                # Run the primary verification tool
                verify_tool = tools[0]
                raw_result = await asyncio.to_thread(
                    verify_tool.invoke, {"tier": "full"}
                )
                if isinstance(raw_result, dict):
                    verify_result = raw_result
                elif isinstance(raw_result, str):
                    verify_result = {
                        "passed": "passed" in raw_result.lower()
                        and "failed" not in raw_result.lower(),
                        "output": raw_result[-4000:],
                    }
        except Exception as exc:
            verify_result = {"passed": False, "error": str(exc)}

        failures: list[str] = []
        if not verify_result.get("passed", False):
            error = verify_result.get("error") or verify_result.get("output") or "Verification failed"
            failures.append(str(error)[:2000])

        return {
            "verify_result": verify_result,
            "failures": failures,
            "graph_stage": "verify_node",
        }

    # ── 9. api_node ────────────────────────────────────────────────────
    async def api_node(state: HarnessGraphState) -> dict[str, Any]:
        """Run API tools (Tavily search, GitHub, Supabase, LangSmith)."""
        api_result: dict[str, Any] = {"status": "complete"}

        try:
            tools = api_tools.get_tools()
            api_result["available_tools"] = [
                getattr(t, "name", str(t)) for t in tools
            ]
        except Exception as exc:
            api_result = {"status": "error", "error": str(exc)}

        return {"api_result": api_result, "graph_stage": "api_node"}

    # ── 10. repair_node ────────────────────────────────────────────────
    async def repair_node(state: HarnessGraphState) -> dict[str, Any]:
        """Prepare repair feedback and advance the repair counter."""
        next_round = state.get("repair_round", 0) + 1
        failures = state.get("failures", [])
        details = "\n".join(f"- {f}" for f in failures)

        feedback = (
            "The previous build round reported failures. Continue working in the same "
            "workspace, diagnose and fix these exact issues, then rerun verification.\n\n"
            f"Failure details:\n{details}\n\n"
            f"Previous output:\n{state.get('build_output', '')[:2000]}"
        )

        return {
            "repair_round": next_round,
            "repair_feedback": feedback,
            "graph_stage": "repair_node",
        }

    # ── 11. session_node ───────────────────────────────────────────────
    async def session_node(state: HarnessGraphState) -> dict[str, Any]:
        """Persist session state (placeholder — harness.run() handles this)."""
        return {"graph_stage": "session_node"}

    # ── 12. output_node ────────────────────────────────────────────────
    async def output_node(state: HarnessGraphState) -> dict[str, Any]:
        """Finalize run status and produce output summary."""
        failures = state.get("failures", [])
        repair_round = state.get("repair_round", 0)
        max_rounds = state.get("max_repair_rounds", max_repair_rounds)

        if state.get("status") == "blocked":
            return {
                "status": "blocked",
                "final_output": state.get("final_output", "Run was blocked."),
                "graph_stage": "output_node",
            }

        test_result = state.get("test_result", {})
        git_result = state.get("git_result", {})
        verify_result = state.get("verify_result", {})

        if verify_result.get("passed") and test_result.get("passed"):
            parts = ["✅ All checks passed."]
            if git_result.get("success"):
                parts.append("Changes pushed to remote.")
            return {
                "status": "complete",
                "final_output": " ".join(parts),
                "graph_stage": "output_node",
            }

        if failures and repair_round >= max_rounds:
            details = "\n".join(f"- {f}" for f in failures)
            return {
                "status": "needs_attention",
                "final_output": (
                    f"{state.get('build_output', '')}\n\n"
                    "⚠️ Automatic repair limit reached before checks passed.\n\n"
                    f"Latest failures:\n{details}"
                ),
                "graph_stage": "output_node",
            }

        return {
            "status": state.get("status", "complete"),
            "final_output": state.get("build_output", "Task completed."),
            "graph_stage": "output_node",
        }

    # ── Routing functions ──────────────────────────────────────────────
    def route_after_guard(state: HarnessGraphState) -> str:
        if not state.get("start_approved", False):
            return "output_node"
        return "planning_node"

    def route_after_verify(state: HarnessGraphState) -> str:
        failures = state.get("failures", [])
        if not failures:
            # Tests passed — proceed to API tools then session/output
            return "api_node"
        repair_round = state.get("repair_round", 0)
        max_rounds = state.get("max_repair_rounds", max_repair_rounds)
        if repair_round >= max_rounds:
            return "output_node"
        return "repair_node"

    # ── Compile the full graph ─────────────────────────────────────────
    graph = StateGraph(HarnessGraphState)

    # Subagent nodes
    graph.add_node("input_guard", input_guard)
    graph.add_node("planning_node", planning_node)
    graph.add_node("design_node", design_node)
    graph.add_node("build_node", build_node)

    # Tool nodes
    graph.add_node("graphrag_node", graphrag_node)
    graph.add_node("tool_pipeline", tool_pipeline)
    graph.add_node("deploy_node", deploy_node)
    graph.add_node("verify_node", verify_node)
    graph.add_node("api_node", api_node)

    # Pipeline nodes
    graph.add_node("repair_node", repair_node)
    graph.add_node("session_node", session_node)
    graph.add_node("output_node", output_node)

    # ── Edges ──────────────────────────────────────────────────────────
    # Entry
    graph.add_edge(START, "input_guard")

    # Subagent chain
    graph.add_conditional_edges("input_guard", route_after_guard)
    graph.add_edge("planning_node", "design_node")
    graph.add_edge("design_node", "graphrag_node")

    # GraphRAG → Build
    graph.add_edge("graphrag_node", "build_node")

    # Build → Tool pipeline → Deploy → Verify
    graph.add_edge("build_node", "tool_pipeline")
    graph.add_edge("tool_pipeline", "deploy_node")
    graph.add_edge("deploy_node", "verify_node")

    # Verify routing: passed → api_node, failed → repair or output
    graph.add_conditional_edges("verify_node", route_after_verify)

    # API → Session → Output → END
    graph.add_edge("api_node", "session_node")
    graph.add_edge("session_node", "output_node")

    # Repair loops back to build
    graph.add_edge("repair_node", "build_node")

    # Terminal
    graph.add_edge("output_node", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Input sanitization — known prompt-injection trigger phrases.
# These are heuristic patterns, not a complete defence; they serve as a
# first-layer trip-wire that logs and blocks the most obvious attacks.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore previous",
        re.compile(
            r"\bignore\s+(?:all\s+)?previous\s+(?:system\s+)?instructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ignore your instructions",
        re.compile(r"\bignore\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "forget your instructions",
        re.compile(r"\bforget\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "disregard your instructions",
        re.compile(r"\bdisregard\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "new instructions:",
        re.compile(
            r"(?im)^\s*new instructions\s*:\s*(?:ignore|disregard|override|reveal)\b"
        ),
    ),
    (
        "system:",
        re.compile(
            r"(?im)^\s*system\s*:\s*(?:ignore|disregard|override|you are now|reveal)\b"
        ),
    ),
    ("<|im_sep|>", re.compile(r"<\|im_sep\|>", re.IGNORECASE)),
    ("<|endoftext|>", re.compile(r"<\|endoftext\|>", re.IGNORECASE)),
    ("<|system|>", re.compile(r"<\|system\|>", re.IGNORECASE)),
    (
        "act as if you",
        re.compile(r"\bact\s+as\s+if\s+you\s+(?:are|were|can|must)\b", re.IGNORECASE),
    ),
    (
        "you are now",
        re.compile(
            r"(?im)^\s*you\s+are\s+now\s+(?:an?\s+)?(?:assistant|system|agent)\b"
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(?:enable|perform|attempt)\s+(?:a\s+)?jailbreak\b", re.IGNORECASE
        ),
    ),
)


# ---------------------------------------------------------------------------
# Supported LLM Models registry
# Includes supported cloud Ollama models.
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "ollama:beam:cloud": {
        "id": "ollama:beam:cloud",
        "provider": "ollama",
        "name": "beam:cloud",
        "label": "Beam (Cloud)",
        "aliases": ["beam:cloud", "beam", "ollama:beam"],
        "fallback": FALLBACK_MODEL,
    },
    "ollama:gemma4:cloud": {
        "id": "ollama:gemma4:cloud",
        "provider": "ollama",
        "name": "gemma4:cloud",
        "label": "Gemma 4 (Cloud)",
        "aliases": ["gemma4:cloud", "gemma4", "ollama:gemma4"],
        "fallback": FALLBACK_MODEL,
    },
    "ollama:nvidia-nemotron-super:cloud": {
        "id": "ollama:nvidia-nemotron-super:cloud",
        "provider": "ollama",
        "name": "nvidia-nemotron-super:cloud",
        "label": "NVIDIA Nemotron Super (Cloud)",
        "aliases": [
            "nvidia-nemotron-super:cloud",
            "nemotron-super:cloud",
            "nemotron-3-super:cloud",
            "ollama:nemotron-3-super:cloud",
        ],
        "fallback": FALLBACK_MODEL,
    },
    FALLBACK_MODEL: {
        "id": FALLBACK_MODEL,
        "provider": "ollama",
        "name": FALLBACK_MODEL.removeprefix("ollama:"),
        "label": "GPT-OSS 120B (Cloud)",
        "aliases": [
            "gpt-oss-120b:cloud",
            FALLBACK_MODEL.removeprefix("ollama:"),
            FALLBACK_MODEL,
        ],
    },
}


def normalize_model_identifier(model_name: str | None) -> str:
    """Normalize user or UI supplied model identifier to registered model key.

    Maps aliases such as 'gemma4:cloud', 'nvidia-nemotron-super:cloud',
    'gpt-oss-120b:cloud', etc., to their canonical representation.
    """
    if not model_name:
        return DEFAULT_MODEL

    cleaned = model_name.strip()
    if cleaned in SUPPORTED_MODELS:
        return cleaned

    # Check aliases
    for canonical_id, config in SUPPORTED_MODELS.items():
        aliases = config.get("aliases", [])
        if cleaned in aliases or cleaned.lower() in [a.lower() for a in aliases]:
            return canonical_id

    # Reject retired providers rather than allowing a stale selection to route externally.
    if ":" in cleaned:
        provider = cleaned.partition(":")[0].lower()
        if provider != "ollama":
            raise ValueError(
                f"Unsupported model provider '{provider}'. Only Ollama models are supported."
            )
        return cleaned
    return f"ollama:{cleaned}"


class CodingAgentHarness:
    """The central harness orchestrator.

    Owns the backend, subagents, and checkpointer.
    Provides run() and resume() thread-scoped entry points.
    """

    SUPPORTED_MODELS = SUPPORTED_MODELS

    def __init__(
        self,
        root_dir: str = ".",
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        tracer: Any = None,
        sandbox_mode: str | None = None,
        runtime: LangGraphRuntime | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._model = normalize_model_identifier(model)
        self._tracer = tracer or LangSmithTracer()
        self._evaluator = HarnessEvaluator(tracer_instance=self._tracer)

        self._guard = HarnessGuard(
            f"{root_dir}/harness/permissions.json",
            silent_mode=True,
            sandbox_mode=sandbox_mode,
        )
        self._backend = DeepAgentsBackend(
            root_dir=root_dir,
            guard=self._guard,
            sandbox_mode=sandbox_mode,
        )
        self._runtime = runtime or LangGraphRuntime(root_dir=root_dir)
        self._sessions = self._runtime.session_manager
        self._execution_graphs: dict[str, Any] = {}
        self._external_tools = ExternalToolsManager()
        self._mcp_clients: list[Any] = []
        self._drawio_client: Any = None
        self._planner_agent: Any = None
        self._design_agent: Any = None
        self._agents_by_model: dict[str, Any] = {}
        self._planners_by_model: dict[str, Any] = {}
        self._loaded_tool_names: list[str] = []
        self._additional_mcp_status: dict[str, str] = {}
        self._deploy_tools = DeployToolset(root_dir=root_dir)
        self._verification_tools = VerificationToolset(root_dir=root_dir)
        self._api_tools = APIToolset()
        self._system_prompt = (
            "You are an autonomous coding harness agent. Always follow AGENTS.md and TASK.md, "
            "respect permissions.json, coordinate planning -> design -> build, and verify all code.\n"
            "When the task requires a shell command or file mutation, invoke the appropriate tool; "
            "do not skip it or ask for approval in chat. HarnessGuard checks the action before it "
            "runs and will pause for the user through the approval UI when the rule is 'ask'. "
            "After approval, continue from the resumed tool result. If denied, do not retry through "
            "another tool or method.\n"
            "For Git, package-install, test, or build commands, use execute_command directly or "
            "delegate to the build-agent. Do not claim that shell or Git access is unavailable "
            "before attempting a tool; let HarnessGuard request approval when required.\n"
            "For code changes, call run_verification after editing. If checks fail, "
            "diagnose and repair the reported issue, then verify again.\n"
            "FILESYSTEM PATH RULES: All filesystem tools (ls, read_file, write_file, edit_file) require "
            "POSIX virtual paths starting with /. Never use Windows paths like C:\\\\path. "
            "The workspace root is /. Use ls(path='/') to explore, read_file(file_path='/agent/core.py'), etc."
        )
        # Build tools list with safety wraps for DB-dependent tools
        try:
            self._tools = (
                list(tools)
                if tools is not None
                else [
                    *self._deploy_tools.get_tools(),
                    *self._verification_tools.get_tools(),
                    *self._api_tools.get_tools(),
                ]
            )
            # B7: build a single shared GraphRAGPipeline so ingest and retrieval use the same
            # store (critical for in-memory fallback mode where they previously shared nothing)
            # and so we only pay the cost of one reranker + embedder load.
            try:
                from graphrag.pipeline import GraphRAGPipeline

                shared_pipeline = GraphRAGPipeline()
                self._tools.append(create_repo_ingest_tool(pipeline=shared_pipeline))
                self._tools.append(
                    create_graphrag_retriever_tool(pipeline=shared_pipeline)
                )
            except Exception as db_exc:  # noqa: BLE001 - optional GraphRAG integration is isolated and logged
                logger.error(f"GraphRAG tools failed to load: {db_exc}")
        except Exception as e:  # noqa: BLE001 - tool initialization failure is logged before degraded startup
            logger.error(f"Base tools initialization failed: {e}")
            self._tools = []

        self._subagents: list[dict[str, Any]] = []
        self._loaded_tool_names = sorted(
            {getattr(tool, "name", str(tool)) for tool in self._tools}
        )
        self._agent: Any = None

    @classmethod
    async def create(
        cls,
        root_dir: str = ".",
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        tracer: Any = None,
        sandbox_mode: str | None = None,
    ) -> CodingAgentHarness:
        """Create a harness and load optional MCP integrations before building agents."""
        sessions = SessionManager(root_dir=root_dir)
        runtime = await LangGraphRuntime.create(
            session_manager=sessions,
            root_dir=root_dir,
        )
        interrupted = sessions.mark_running_interrupted()
        if interrupted:
            logger.warning(
                "Recovered %d interrupted run(s) from session storage.",
                len(interrupted),
            )
        harness = cls(
            root_dir=root_dir,
            model=model,
            tools=tools,
            tracer=tracer,
            sandbox_mode=sandbox_mode,
            runtime=runtime,
        )
        try:
            (
                build_tools,
                external_tools,
                mcp_clients,
            ) = await get_build_dev_tools(
                base_tools=harness._tools,
                external_tools_manager=harness._external_tools,
                root_dir=root_dir,
                guard=harness._guard,
                backend=harness._backend.backend,
            )
            planner = make_planning_subagent(
                root_dir=root_dir,
                guard=harness._guard,
                model=harness._model,
            )
            designer, drawio_client = await make_design_subagent(
                root_dir=root_dir,
                guard=harness._guard,
                model=harness._model,
            )
            harness._planner_agent = planner
            harness._design_agent = designer
            harness._drawio_client = drawio_client
            harness._additional_mcp_status = {
                "drawio": (
                    f"loaded {len(drawio_client.tool_names)} tools"
                    if drawio_client.tool_names
                    else "unavailable"
                )
            }
            harness._external_tools = external_tools
            harness._mcp_clients = mcp_clients
            harness._subagents = [
                {
                    "name": "planning-agent",
                    "description": planning_subagent["description"],
                    "runnable": planner,
                },
                {
                    "name": "design-agent",
                    "description": design_subagent["description"],
                    "runnable": designer,
                },
                build_dev_subagent(
                    build_tools,
                    external_tools=external_tools,
                    mcp_clients=mcp_clients,
                    root_dir=root_dir,
                    backend=harness._backend.backend,
                    guard=harness._guard,
                    tools_already_guarded=True,
                    model=harness._model,
                ),
            ]
            harness._loaded_tool_names = sorted(
                set(harness._loaded_tool_names)
                | set(external_tools.loaded_tool_names)
                | set(drawio_client.tool_names)
                | {getattr(tool, "name", str(tool)) for tool in build_tools}
            )
            harness._agent = harness._build_root_agent(harness._model)
            harness._agents_by_model[harness._model] = harness._agent
            harness._planners_by_model[harness._model] = planner
            logger.info("Harness initialized with model %s for root and subagents.", harness._model)
            return harness
        except Exception:
            await harness.aclose()
            raise

    def _build_root_agent(
        self, model: str, subagents: list[dict[str, Any]] | None = None
    ) -> Any:
        return self._backend.build_agent(
            model=model,
            subagents=self._subagents if subagents is None else subagents,
            system_prompt=self._system_prompt,
            tools=self._get_guarded_tools(),
            checkpointer=self._runtime.checkpointer,
            state_schema=AgentState,
        )

    def _get_guarded_tools(self) -> list[Any]:
        return [
            self._guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt)
            for tool in self._tools
        ]

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError(
                "Initialize the harness with `await CodingAgentHarness.create(...)`."
            )
        return self._agent

    @property
    def guard(self) -> HarnessGuard:
        return self._guard

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def sessions(self) -> SessionManager:
        return self._sessions

    @property
    def runtime(self) -> LangGraphRuntime:
        return self._runtime

    @property
    def tracer(self) -> LangSmithTracer:
        return self._tracer

    @property
    def evaluator(self) -> HarnessEvaluator:
        return self._evaluator

    @property
    def model(self) -> str:
        """Return the default active model identifier."""
        return self._model

    @property
    def supported_models(self) -> dict[str, dict[str, Any]]:
        """Return dictionary of supported models."""
        return self.SUPPORTED_MODELS

    @property
    def loaded_tool_names(self) -> list[str]:
        return list(self._loaded_tool_names)

    @property
    def mcp_server_status(self) -> dict[str, str]:
        return {**self._external_tools.server_status, **self._additional_mcp_status}

    def create_repair_graph(
        self,
        agent: Any | None = None,
        *,
        max_repair_rounds: int = 3,
        pre_run: Any = None,
        planner: Any = None,
    ) -> Any:
        """Create a bounded repair controller around an agent graph."""
        selected_agent = agent if agent is not None else self._agent
        if selected_agent is None:
            raise RuntimeError(
                "Initialize the harness with `await CodingAgentHarness.create(...)`."
            )
        return build_repair_graph(
            selected_agent,
            checkpointer=self._runtime.checkpointer,
            max_repair_rounds=max_repair_rounds,
            guard=self._guard,
            pre_run=pre_run,
            planner=planner
            or self._planner_agent
            or make_planning_subagent(
                root_dir=self._root_dir,
                guard=self._guard,
                model=self._model,
            ),
            root_dir=self._root_dir,
        )

    def build_pipeline_graph(
        self,
        *,
        max_repair_rounds: int = 3,
    ) -> Any:
        """Build the full harness pipeline graph connecting all components.

        The pipeline graph wires every agent component into a single connected
        LangGraph StateGraph::

            input_guard → planning → design → graphrag → build
                → tool_pipeline (subgraph: grep/glob → read → write → edit → pytest → git push)
                → deploy → verify → [api | repair → build (loop)] → session → output

        Returns:
            A compiled ``StateGraph`` ready for ``ainvoke()``.

        Raises:
            RuntimeError: If the harness has not been initialized via ``create()``.
        """
        if not self._subagents or len(self._subagents) < 3:
            raise RuntimeError(
                "Initialize the harness with `await CodingAgentHarness.create(...)` "
                "before building the pipeline graph."
            )

        # Resolve the GraphRAG pipeline (may be None if DB is unavailable)
        graphrag_pipeline = None
        try:
            from graphrag.pipeline import GraphRAGPipeline
            graphrag_pipeline = GraphRAGPipeline()
        except Exception:
            logger.debug("GraphRAG pipeline unavailable for pipeline graph.")

        return build_harness_graph(
            planner_agent=self._planner_agent,
            design_agent=self._design_agent,
            build_agent=self._subagents[2],  # build-agent spec dict
            backend=self._backend.backend,
            guard=self._guard,
            deploy_tools=self._deploy_tools,
            verification_tools=self._verification_tools,
            api_tools=self._api_tools,
            graphrag_pipeline=graphrag_pipeline,
            root_dir=self._root_dir,
            max_repair_rounds=max_repair_rounds,
            checkpointer=self._runtime.checkpointer,
        )

    async def get_agent_for_model(self, model: str) -> Any:
        """Return a root graph whose root and all delegated agents use this model."""
        chosen_model = normalize_model_identifier(model)
        cached_agent = self._agents_by_model.get(chosen_model)
        if cached_agent is not None:
            return cached_agent
        if len(self._subagents) < 3:
            raise RuntimeError("Harness subagents are not initialized.")

        planner = make_planning_subagent(
            root_dir=self._root_dir,
            guard=self._guard,
            model=chosen_model,
        )
        designer, _ = await make_design_subagent(
            root_dir=self._root_dir,
            guard=self._guard,
            model=chosen_model,
            drawio_client=self._drawio_client,
        )
        model_subagents = [
            {**self._subagents[0], "runnable": planner},
            {**self._subagents[1], "runnable": designer},
            {**self._subagents[2], "model": chosen_model},
        ]
        agent = self._build_root_agent(chosen_model, subagents=model_subagents)
        self._agents_by_model[chosen_model] = agent
        self._planners_by_model[chosen_model] = planner
        logger.info("Built root and subagents with model %s.", chosen_model)
        return agent

    def build_agent_for_model(self, model: str) -> Any:
        """Build the full planning/design/build agent graph for a model."""
        chosen_model = normalize_model_identifier(model)
        if chosen_model == self._model:
            if self._agent is None:
                raise RuntimeError(
                    "Initialize the harness with `await CodingAgentHarness.create(...)`."
                )
            return self._agent

        return self._build_root_agent(chosen_model)

    def _conversation_history(self, thread_id: str, user_request: str) -> str:
        """Format prior persisted turns, excluding the request currently being run."""
        messages = self._sessions.get_messages(thread_id)
        if (
            messages
            and messages[-1].get("sender") == "user"
            and messages[-1].get("text") == user_request
        ):
            messages = messages[:-1]
        recent_messages = messages[-20:]
        return "\n".join(
            f"{message.get('sender', 'unknown').capitalize()}: {message.get('text', '')}"
            for message in recent_messages
            if message.get("text")
        )

    @property
    def skills(self):
        """Access the global SkillLibrary to register or load custom skills.

        Example::

            from skills.schema import Skill
            harness.skills.register(Skill(
                name="my-convention",
                description="Team coding conventions",
                tags=["python", "style"],
                instructions="Always use Black. Max line 88. Type-annotate all functions.",
            ))
            harness.skills.load_from_directory("./my_skills/")
        """
        return skill_library

    # ------------------------------------------------------------------
    # Input sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_suspicious_input(text: str) -> str | None:
        """Return a narrowly matched instruction pattern, ignoring code literals."""
        prose = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
        prose = re.sub(r"`[^`]*`", "", prose)
        for label, pattern in _INJECTION_PATTERNS:
            if pattern.search(prose):
                return label
        return None

    async def run(
        self, user_request: str, thread_id: str, model: str | None = None
    ) -> dict[str, Any]:
        """Executes a run for a specific thread with an optional model override.

        Performs a prompt-injection pre-flight check before handing the request
        to the agent. Suspicious inputs are logged to harness/failures.md and
        returned immediately as ``{"status": "blocked"}`` without reaching the
        agent graph.
        """
        # ── Pre-flight: prompt-injection guard ─────────────────────────────────
        matched = self._is_suspicious_input(user_request)
        if matched:
            preview = user_request[:120].replace("\n", " ")
            self._guard.log_failure(
                stage="input",
                category="prompt-injection",
                description=f"Matched pattern '{matched}' in: {preview}",
                resolution="Request blocked before reaching agent.",
            )
            if not self._sessions.has_session(thread_id):
                title_preview = user_request.strip().split("\n")[0][:40]
                self._sessions.create_session(
                    title=title_preview or f"Session {thread_id[:8]}",
                    session_id=thread_id,
                )
            self._sessions.update_session(thread_id, status="blocked")
            await self._runtime.persist_session_to_memory(thread_id)
            return {
                "status": "blocked",
                "reason": (
                    f"Input flagged as potential prompt injection "
                    f"(matched pattern: '{matched}'). Request not forwarded to agent."
                ),
            }

        config = {"configurable": {"thread_id": thread_id}}

        # Track or register the session in SessionManager
        if not self._sessions.has_session(thread_id):
            # Derive title from first request preview
            title_preview = user_request.strip().split("\n")[0][:40]
            self._sessions.create_session(
                title=title_preview or f"Session {thread_id[:8]}", session_id=thread_id
            )

        messages = self._sessions.get_messages(thread_id)
        if not messages or (
            messages[-1].get("sender") != "user"
            or messages[-1].get("text") != user_request
        ):
            self._sessions.save_message(thread_id, sender="user", text=user_request)
        self._sessions.update_session(thread_id, status="running")
        await self._runtime.persist_session_to_memory(thread_id)

        chosen_model = normalize_model_identifier(model) if model else self._model
        logger.info("Starting run with model %s.", chosen_model)
        agent_to_invoke = await self.get_agent_for_model(chosen_model)

        async def prepare_agent_run(state: dict[str, Any]) -> None:
            await generate_agents_md(
                user_request=state["user_request"],
                root_dir=self._root_dir,
                model=chosen_model,
            )

        execution_graph = self.create_repair_graph(
            agent_to_invoke,
            pre_run=prepare_agent_run,
            planner=self._planners_by_model[chosen_model],
        )
        self._execution_graphs[thread_id] = execution_graph
        try:
            result = await execution_graph.ainvoke(
                {
                    "user_request": user_request,
                    "conversation_history": self._conversation_history(
                        thread_id, user_request
                    ),
                    "execution_thread_id": f"{thread_id}:execution:{uuid.uuid4()}",
                    "model_id": chosen_model,
                    "repair_round": 0,
                    "max_repair_rounds": 3,
                },
                config=config,
            )
        except asyncio.CancelledError:
            self._execution_graphs.pop(thread_id, None)
            self._sessions.update_session(thread_id, status="interrupted")
            await self._runtime.persist_session_to_memory(thread_id)
            raise
        except Exception:
            self._sessions.update_session(thread_id, status="failed")
            await self._runtime.persist_session_to_memory(thread_id)
            self._execution_graphs.pop(thread_id, None)
            raise

        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        final_status = result.get("status", "complete")
        self._sessions.update_session(thread_id, status=final_status)
        await self._runtime.persist_session_to_memory(thread_id)
        self._execution_graphs.pop(thread_id, None)
        return {
            "status": final_status,
            "output": result.get("final_output", "Task completed."),
            "result": result,
        }

    async def resume(
        self, thread_id: str, approved: bool | None = None
    ) -> dict[str, Any]:
        """Resume a pending approval or recover a checkpointed run after restart."""
        config = {"configurable": {"thread_id": thread_id}}
        graph = self._execution_graphs.get(thread_id)
        try:
            if graph is None:
                graph = self.create_repair_graph(agent=self._agent)
                snapshot = await graph.aget_state(config)
                if snapshot is not None and snapshot.values:
                    chosen_model = normalize_model_identifier(
                        snapshot.values.get("model_id", self._model)
                    )
                    if chosen_model == self._model:
                        agent = self._agent
                        planner = self._planner_agent
                    else:
                        agent = await self.get_agent_for_model(chosen_model)
                        planner = self._planners_by_model[chosen_model]

                    async def prepare_recovered_run(state: dict[str, Any]) -> None:
                        await generate_agents_md(
                            user_request=state["user_request"],
                            root_dir=self._root_dir,
                            model=chosen_model,
                        )

                    graph = self.create_repair_graph(
                        agent=agent,
                        pre_run=prepare_recovered_run,
                        planner=planner,
                    )
                self._execution_graphs[thread_id] = graph

            snapshot = await graph.aget_state(config)
            if snapshot is None or not snapshot.values:
                raise ValueError(f"No checkpointed run exists for thread {thread_id}.")

            self._sessions.update_session(thread_id, status="running")
            await self._runtime.persist_session_to_memory(thread_id)
            if approved is None:
                result = await graph.ainvoke(None, config=config)
            else:
                result = await graph.ainvoke(
                    Command(resume={"approved": approved}),
                    config=config,
                )
        except asyncio.CancelledError:
            self._execution_graphs.pop(thread_id, None)
            self._sessions.update_session(thread_id, status="interrupted")
            await self._runtime.persist_session_to_memory(thread_id)
            raise
        except Exception:
            logger.exception("Resume execution failed for thread %s; stopping the run.", thread_id)
            self._execution_graphs.pop(thread_id, None)
            self._sessions.update_session(thread_id, status="failed")
            await self._runtime.persist_session_to_memory(thread_id)
            raise
        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        final_status = (
            result.get("status", "complete") if isinstance(result, dict) else "complete"
        )
        self._sessions.update_session(thread_id, status=final_status)
        await self._runtime.persist_session_to_memory(thread_id)
        self._execution_graphs.pop(thread_id, None)
        return {
            "status": final_status,
            "output": result.get("final_output", "Task completed.")
            if isinstance(result, dict)
            else "Task completed.",
            "result": result,
        }

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Retrieves the current state of a thread from checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        graph = self._execution_graphs.get(thread_id)
        if graph is None:
            graph = self.create_repair_graph(agent=self._agent)
        snapshot = await graph.aget_state(config)
        return snapshot.values if snapshot else None

    async def stream(self, thread_id: str) -> AsyncIterator[Any]:
        """Streams updates for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self._agent.astream(None, config=config):
            yield chunk

    async def aclose(self) -> None:
        """Close every MCP and database session owned by the harness."""
        if self._drawio_client is not None:
            try:
                await self._drawio_client.close()
            except Exception as exc:
                logger.warning("Failed to close an MCP session: %s", exc)
        for client in reversed(self._mcp_clients):
            try:
                await client.close()
            except Exception as exc:
                logger.warning("Failed to close an MCP session: %s", exc)
        await self._external_tools.close_all()
        await self._runtime.aclose()
