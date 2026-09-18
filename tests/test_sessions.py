"""
tests/test_sessions.py - Unit test verifying multi-session management and LangGraphRuntime integration.
"""

import pytest
import shutil
from pathlib import Path
from agent.session_manager import SessionManager
from frameworks.langgraph_runtime import LangGraphRuntime


def test_session_lifecycle(tmp_path: Path):
    manager = SessionManager(root_dir=str(tmp_path))

    # 1. Create sessions
    s1 = manager.create_session("Build auth feature")
    s2 = manager.create_session("Refactor database schema")

    assert s1.id != s2.id
    assert s1.title == "Build auth feature"
    assert s2.title == "Refactor database schema"
    assert manager.has_session(s1.id)
    assert manager.has_session(s2.id)

    # 2. List sessions
    sessions = manager.list_sessions()
    assert len(sessions) == 2

    # 3. Update session
    manager.update_session(s1.id, stage="design", increment_messages=True)
    updated = manager.get_session(s1.id)
    assert updated is not None
    assert updated.stage == "design"
    assert updated.message_count == 1

    # 4. Persistence across manager instances
    manager_reloaded = SessionManager(root_dir=str(tmp_path))
    reloaded_s1 = manager_reloaded.get_session(s1.id)
    assert reloaded_s1 is not None
    assert reloaded_s1.title == "Build auth feature"

    # 5. Delete session
    assert manager.delete_session(s2.id) is True
    assert manager.get_session(s2.id) is None
    assert len(manager.list_sessions()) == 1


def test_langgraph_runtime_session_integration(tmp_path: Path):
    manager = SessionManager(root_dir=str(tmp_path))
    runtime = LangGraphRuntime(session_manager=manager, root_dir=str(tmp_path))

    # Test create_session via ensure_session
    thread_id = "thread-xyz-123"
    session = runtime.ensure_session(thread_id, title="Test LangGraph Session")
    assert session.id == thread_id
    assert session.title == "Test LangGraph Session"
    assert session.status == "active"
    assert session.message_count == 0

    # Ensure calling again returns the existing session
    session_dup = runtime.ensure_session(thread_id, title="Another Title")
    assert session_dup.id == thread_id
    assert session_dup.title == "Test LangGraph Session"

    # Test get_thread_config
    config = runtime.get_thread_config(thread_id)
    assert config == {"configurable": {"thread_id": thread_id}}

    # Test update_session via update_session_state
    updated = runtime.update_session_state(
        thread_id,
        stage="build",
        status="awaiting_approval",
        increment_messages=True,
    )
    assert updated is not None
    assert updated.stage == "build"
    assert updated.status == "awaiting_approval"
    assert updated.message_count == 1

    # Verify session list
    sessions = runtime.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].id == thread_id
    assert sessions[0].stage == "build"
