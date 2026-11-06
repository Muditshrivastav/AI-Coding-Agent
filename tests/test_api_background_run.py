import asyncio

import pytest

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


def test_failed_run_resume_is_rejected_without_restarting(monkeypatch):
    class FakeSessions:
        def get_session(self, thread_id):
            return type("Session", (), {"status": "failed"})()

    class FakeHarness:
        def __init__(self):
            self.sessions = FakeSessions()

    async def exercise():
        fake_harness = FakeHarness()
        monkeypatch.setattr(api_server, "harness", fake_harness)
        api_server.active_run_tasks.clear()

        from fastapi import HTTPException

        for _ in range(2):
            with pytest.raises(HTTPException) as exc_info:
                await api_server.resume_run(
                    "failed-thread",
                    api_server.ResumeRequest(approved=True),
                )
            assert exc_info.value.status_code == 409
            assert "start a new run" in exc_info.value.detail

    asyncio.run(exercise())


def test_resume_errors_are_recorded_in_failure_log(monkeypatch, tmp_path):
    class FakeSessions:
        def get_session(self, thread_id):
            return type("Session", (), {"status": "awaiting_approval"})()

    class FakeGuard:
        def __init__(self):
            self.logged = []

        def log_failure(self, **kwargs):
            self.logged.append(kwargs)

    class FakeHarness:
        def __init__(self):
            self.sessions = FakeSessions()
            self._guard = FakeGuard()

        async def resume(self, thread_id, *, approved):
            raise RuntimeError("Checkpoint store unavailable")

    async def exercise():
        fake_harness = FakeHarness()
        monkeypatch.setattr(api_server, "harness", fake_harness)
        api_server.active_run_tasks.clear()

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await api_server.resume_run(
                "broken-thread",
                api_server.ResumeRequest(approved=True),
            )

        assert exc_info.value.status_code == 500
        assert fake_harness._guard.logged == [
            {
                "stage": "resume",
                "category": "RuntimeError",
                "description": "Thread broken-thread: Checkpoint store unavailable",
                "resolution": (
                    "Resume stopped and the session was marked failed; start a new run."
                ),
            }
        ]

    asyncio.run(exercise())
