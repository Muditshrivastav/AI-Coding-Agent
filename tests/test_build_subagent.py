"""
tests/test_build_subagent.py - Unit tests for build subagent tools and MCP integration.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from nodes.build_subagent import (
    build_dev_subagent,
    build_subagent,
    get_build_dev_tools,
    make_build_subagent,
    BUILD_SYSTEM_PROMPT,
)
from frameworks.external_tools import ExternalToolsManager
from frameworks.mcp_client import MCPClientWrapper, RenderMCPClient


def test_build_dev_subagent_spec():
    ext_mgr = ExternalToolsManager()
    spec = build_dev_subagent(all_tools=[], external_tools=ext_mgr)

    assert spec["name"] == "build-agent"
    assert "tools" in spec
    # Should include Tavily tool from external tools manager synchronously
    assert any("tavily" in getattr(t, "name", "").lower() for t in spec["tools"])
    # Should include programmatic run_verification tool synchronously
    assert any(getattr(t, "name", "") == "run_verification" for t in spec["tools"])
    # Should include LocalShellBackend execute_command tool synchronously
    assert any(getattr(t, "name", "") == "execute_command" for t in spec["tools"])
    assert spec["factory"] == make_build_subagent
    assert spec["external_tools"] is ext_mgr


def test_build_subagent_guards_prebuilt_tools(tmp_path):
    import json
    from agent.guardrail import HarnessGuard

    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    permissions_path = harness_dir / "permissions.json"
    permissions_path.write_text(
        json.dumps({"allow": ["custom_mutation"], "ask": [], "deny": []}),
        encoding="utf-8",
    )

    calls = []

    def custom_mutation(value: str) -> str:
        calls.append(value)
        return value

    raw_tool = StructuredTool.from_function(
        func=custom_mutation,
        name="custom_mutation",
        description="A prebuilt mutation tool",
    )
    external_tools = ExternalToolsManager()
    external_tools._tavily_tool = None
    guard = HarnessGuard(str(permissions_path))
    spec = build_dev_subagent(
        all_tools=[raw_tool],
        external_tools=external_tools,
        root_dir=str(tmp_path),
        guard=guard,
    )
    guarded_tool = next(tool for tool in spec["tools"] if tool.name == "custom_mutation")

    assert guarded_tool is not raw_tool
    assert guarded_tool.invoke({"value": "approved"}) == "approved"
    assert calls == ["approved"]


def test_probabilistic_decision_uses_ollama_nimble(monkeypatch):
    from langchain_core.messages import AIMessage
    from nodes.build_subagent import create_probabilistic_decision_tool

    calls = {}

    class FakeDecisionModel:
        def invoke(self, prompt):
            calls["prompt"] = prompt
            return AIMessage(content="Recommended action: investigate.")

    def create_model(**kwargs):
        calls["model_kwargs"] = kwargs
        return FakeDecisionModel()

    monkeypatch.setattr("nodes.build_subagent.ChatOllama", create_model)
    tool = create_probabilistic_decision_tool()

    result = tool.invoke(
        {"state": "Deployment returned HTTP 500.", "decision_type": "strategy"}
    )

    assert calls["model_kwargs"]["model"] == "nimble:latest"
    assert "investigate, retry, rollback, or escalate" in calls["prompt"]
    assert "Deployment returned HTTP 500." in calls["prompt"]
    assert "Recommended action: investigate." in result


def test_get_build_dev_tools():
    import asyncio

    async def _test():
        ext_mgr = ExternalToolsManager()
        dummy_mcp_tool = StructuredTool.from_function(
            func=lambda x: x,
            name="mcp__github__create_pull_request",
            description="Create PR",
        )
        mock_client = MagicMock(spec=MCPClientWrapper)
        mock_client.get_tools = AsyncMock(return_value=[dummy_mcp_tool])

        with patch.object(ext_mgr, "get_tools", AsyncMock(return_value=[dummy_mcp_tool])), \
             patch.dict("os.environ", {"RENDER_API_KEY": ""}, clear=False):
            tools, returned_mgr, returned_clients = await get_build_dev_tools(
                base_tools=[],
                external_tools_manager=ext_mgr,
                mcp_clients=[mock_client],
                include_deploy=False,
            )

            assert any(getattr(t, "name", "") == dummy_mcp_tool.name for t in tools)
            assert any(getattr(t, "name", "") == "run_verification" for t in tools)
            assert any(getattr(t, "name", "") == "execute_command" for t in tools)
            assert returned_mgr is ext_mgr
            assert mock_client in returned_clients

    asyncio.run(_test())


def test_execute_command_hitl_guardrail(tmp_path):
    from nodes.build_subagent import create_shell_tool
    from deepagents.backends import LocalShellBackend
    from agent.guardrail import HarnessGuard

    mock_backend = MagicMock(spec=LocalShellBackend)
    mock_backend.execute.return_value = MagicMock(exit_code=0, output="git version 2.40.0")

    guard = HarnessGuard(
        "harness/permissions.json",
        failures_path=str(tmp_path / "failures.md"),
    )
    tool = create_shell_tool(backend=mock_backend, guard=guard)
    run_shell = tool.func
    assert run_shell is not None

    # 1. Even routine commands require approval under the current policy.
    with patch("nodes.build_subagent.interrupt", return_value={"approved": True}) as mock_interrupt:
        res_approved = run_shell("git status", timeout_seconds=1200)
        mock_interrupt.assert_called_once()
    assert "succeeded" in res_approved
    mock_backend.execute.assert_called_with("git status", timeout=1200)

    # 2. Test denied dangerous command
    res_deny = run_shell("rm -rf /")
    assert "SECURITY DENIED" in res_deny

    # 3. Test ask command with interrupt simulation
    with patch("nodes.build_subagent.interrupt", return_value={"approved": True}) as mock_interrupt:
        res_ask_approved = run_shell("git push origin main")
        mock_interrupt.assert_called_once()
        assert "succeeded" in res_ask_approved

    with patch("nodes.build_subagent.interrupt", return_value={"approved": False}) as mock_interrupt:
        res_ask_rejected = run_shell("git push origin main")
        assert "HITL REJECTED" in res_ask_rejected


def test_execute_command_pauses_then_runs_after_graph_approval(tmp_path):
    import asyncio
    import json
    from typing import TypedDict
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command
    from agent.guardrail import HarnessGuard
    from nodes.build_subagent import create_shell_tool

    class ShellState(TypedDict, total=False):
        result: str

    permissions_path = tmp_path / "permissions.json"
    permissions_path.write_text(
        json.dumps({"allow": [], "ask": ["Bash(git status*)"], "deny": []}),
        encoding="utf-8",
    )
    backend = MagicMock(spec=LocalShellBackend)
    backend.execute.return_value = MagicMock(exit_code=0, output="status completed")
    guard = HarnessGuard(
        str(permissions_path),
        failures_path=str(tmp_path / "failures.md"),
    )
    tool = create_shell_tool(backend, guard=guard)

    graph_builder = StateGraph(ShellState)
    graph_builder.add_node(
        "command",
        lambda state: {"result": tool.invoke({"cmd": "git status"})},
    )
    graph_builder.add_edge(START, "command")
    graph_builder.add_edge("command", END)
    config: RunnableConfig = {"configurable": {"thread_id": "approval-resume-test"}}
    graph = graph_builder.compile(checkpointer=MemorySaver())

    async def run_and_approve():
        paused = await graph.ainvoke({}, config=config)
        assert paused.get("__interrupt__")
        backend.execute.assert_not_called()

        resumed = await graph.ainvoke(
            Command(resume={"approved": True}),
            config=config,
        )
        assert "succeeded" in resumed["result"]
        backend.execute.assert_called_once_with("git status", timeout=900)

    asyncio.run(run_and_approve())


def test_workspace_shell_inherits_path_and_runs_from_workspace(tmp_path):
    from deepagents.backends import LocalShellBackend

    backend = LocalShellBackend(
        root_dir=tmp_path,
        timeout=900,
        inherit_env=True,
    )

    assert backend.cwd == tmp_path
    assert backend._env.get("PATH")
    response = backend.execute("python --version", timeout=30)
    assert response.exit_code == 0
    assert "Python" in response.output


def test_real_mcp_server_integration(tmp_path):
    """Integration test verifying end-to-end communication with an authentic real MCP server.

    Launches a real Python MCP server via stdio transport and verifies that
    MCPClientWrapper connects, discovers tools, and get_build_dev_tools integrates them.
    """
    import asyncio
    import sys
    import textwrap

    # 1. Create a real, runnable MCP server script compatible with mcp 2.x & fastmcp
    server_script = tmp_path / "real_server.py"
    server_script.write_text(
        textwrap.dedent("""
            try:
                from fastmcp import FastMCP
                server = FastMCP("RealTestServer")
            except ImportError:
                try:
                    from mcp.server.mcpserver import MCPServer
                    server = MCPServer("RealTestServer")
                except ImportError:
                    from mcp.server.fastmcp import FastMCP
                    server = FastMCP("RealTestServer")

            @server.tool()
            def add_numbers(a: int, b: int):
                '''Adds two numbers together.'''
                return a + b

            if __name__ == "__main__":
                server.run(transport="stdio")
        """),
        encoding="utf-8",
    )

    async def _test():
        # 2. Instantiate a real MCPClientWrapper connecting to this real server via stdio
        real_client = MCPClientWrapper(
            transport="stdio",
            command=sys.executable,
            args=[str(server_script)],
        )

        try:
            # Connect to real server and fetch real tools
            await real_client.connect()
            real_tools = await real_client.get_tools()

            # Verify the tool was retrieved from the actual MCP server
            assert any(t.name == "add_numbers" for t in real_tools)

            # 3. Integrate this real MCP client into get_build_dev_tools
            ext_mgr = ExternalToolsManager()
            tools, returned_mgr, returned_clients = await get_build_dev_tools(
                base_tools=[],
                external_tools_manager=ext_mgr,
                mcp_clients=[real_client],
                include_deploy=False,
            )

            # Verify the real tool is registered and protected by the harness guard
            assert any(t.name == "add_numbers" for t in tools)
            assert real_client in returned_clients
        finally:
            try:
                await real_client.close()
            except Exception:
                pass

    asyncio.run(_test())
