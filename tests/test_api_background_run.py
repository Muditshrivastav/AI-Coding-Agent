import asyncio

from agent.session_manager import SessionManager
from interfaces import api_server


def test_start_run_returns_before_background_execution_finishes(monkeypatch, tmp_path):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        sessions = SessionManager(root_dir=str(tmp_path))

        class FakeHarness:
            def __init__(self):
                self.sessions = sessions

            async def run(self, user_request, *, thread_id, model):
                started.set()
                await release.wait()
                return {"status": "complete", "output": "Finished."}

        monkeypatch.setattr(api_server, "harness", FakeHarness())
        api_server.active_run_tasks.clear()

        response = await api_server.start_run(
            api_server.RunRequest(
                user_request="Implement a small change",
                thread_id="background-thread",
            )
        )
        assert response == {"thread_id": "background-thread", "status": "running"}

        task = api_server.active_run_tasks["background-thread"]
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()

        release.set()
        await asyncio.wait_for(task, timeout=1)
        assert sessions.get_messages("background-thread")[-1]["text"] == "Finished."
        assert "background-thread" not in api_server.active_run_tasks

    asyncio.run(exercise())


def test_mcp_server_failure_does_not_hide_available_tools(monkeypatch):
    from langchain_core.tools import StructuredTool
    from frameworks.external_tools import ExternalToolsManager

    class FakeClient:
        def __init__(self, tool=None, error=None):
            self.tool = tool
            self.error = error

        async def get_tools(self):
            if self.error:
                raise self.error
            return [self.tool]

    async def exercise():
        manager = ExternalToolsManager()
        working_tool = StructuredTool.from_function(
            func=lambda: "ok",
            name="working_mcp_tool",
            description="A working MCP tool",
        )
        manager._mcp_clients = {
            "down": FakeClient(error=RuntimeError("server unavailable")),
            "working": FakeClient(tool=working_tool),
        }
        monkeypatch.setenv("MCP_SERVER_TIMEOUT_SECONDS", "1")

        tools = await manager.get_tools(include_mcp=True, include_search=False)

        assert [tool.name for tool in tools] == ["working_mcp_tool"]
        assert "unavailable" in manager.server_status["down"]
        assert manager.server_status["working"] == "loaded 1 tools"

    asyncio.run(exercise())
