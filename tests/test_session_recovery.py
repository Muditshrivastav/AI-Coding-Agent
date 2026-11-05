import asyncio
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from agent.core import CodingAgentHarness
from agent.session_manager import SessionManager
from frameworks.langgraph_runtime import LangGraphRuntime


def test_running_sessions_are_marked_interrupted_after_restart(tmp_path):
    sessions = SessionManager(root_dir=str(tmp_path))
    sessions.create_session(title="Recover me", session_id="recoverable")
    sessions.update_session("recoverable", status="running")

    restarted_sessions = SessionManager(root_dir=str(tmp_path))
    assert restarted_sessions.mark_running_interrupted() == ["recoverable"]

    loaded_sessions = SessionManager(root_dir=str(tmp_path))
    assert loaded_sessions.get_session("recoverable").status == "interrupted"


def test_runtime_uses_memory_checkpointer_without_postgres_dsn(monkeypatch, tmp_path):
    for key in ("supabase_uri", "SUPABASE_URI", "DATABASE_URL", "SUPABASE_DB_URL"):
        monkeypatch.delenv(key, raising=False)

    sessions = SessionManager(root_dir=str(tmp_path))
    runtime = asyncio.run(
        LangGraphRuntime.create(session_manager=sessions, root_dir=str(tmp_path))
    )
    try:
        assert isinstance(runtime.checkpointer, MemorySaver)
    finally:
        asyncio.run(runtime.aclose())


def test_runtime_uses_async_postgres_checkpointer_when_dsn_exists(
    monkeypatch, tmp_path
):
    from langgraph.checkpoint.postgres import aio

    calls = {}
    for key in ("supabase_uri", "SUPABASE_URI", "SUPABASE_DB_URL"):
        monkeypatch.delenv(key, raising=False)

    class FakeContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *exc_info):
            calls["closed"] = True

    class FakePostgresSaver:
        @classmethod
        def from_conn_string(cls, dsn):
            calls["dsn"] = dsn
            return FakeContext()

    class FakeSaver:
        async def setup(self):
            calls["setup"] = True

    saver = FakeSaver()
    monkeypatch.setattr(aio, "AsyncPostgresSaver", FakePostgresSaver)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    sessions = SessionManager(root_dir=str(tmp_path))

    runtime = asyncio.run(
        LangGraphRuntime.create(session_manager=sessions, root_dir=str(tmp_path))
    )
    try:
        assert runtime.checkpointer is saver
        assert calls == {"dsn": "postgresql://test", "setup": True}
    finally:
        asyncio.run(runtime.aclose())
    assert calls["closed"] is True


def test_resume_rebuilds_execution_graph_after_restart(monkeypatch, tmp_path):
    calls = {}

    class FakeGraph:
        async def aget_state(self, config):
            return SimpleNamespace(values={"status": "running"})

        async def ainvoke(self, input_value, *, config):
            calls["input"] = input_value
            calls["config"] = config
            return {"status": "complete", "final_output": "Recovered."}

    class FakeSessions:
        def update_session(self, thread_id, *, status):
            calls["session_status"] = status

    class FakeRuntime:
        async def persist_session_to_memory(self, thread_id):
            calls["persisted"] = thread_id

    harness = CodingAgentHarness.__new__(CodingAgentHarness)
    harness._execution_graphs = {}
    harness._agent = object()
    harness._root_dir = str(tmp_path)
    harness._model = "test-model"
    harness._sessions = FakeSessions()
    harness._runtime = FakeRuntime()
    graph = FakeGraph()

    def rebuild(*, agent, pre_run):
        calls["agent"] = agent
        calls["pre_run"] = pre_run
        return graph

    monkeypatch.setattr(harness, "create_repair_graph", rebuild)

    result = asyncio.run(harness.resume("recoverable", approved=None))

    assert result == {
        "status": "complete",
        "output": "Recovered.",
        "result": {"status": "complete", "final_output": "Recovered."},
    }
    assert calls["agent"] is harness._agent
    assert calls["pre_run"] is not None
    assert calls["input"] is None
    assert calls["config"] == {"configurable": {"thread_id": "recoverable"}}
    assert calls["session_status"] == "complete"
    assert calls["persisted"] == "recoverable"
    assert not harness._execution_graphs
