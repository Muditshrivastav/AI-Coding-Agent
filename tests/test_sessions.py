"""
tests/test_sessions.py - Unit test verifying multi-session management.
"""

import shutil
from pathlib import Path
from agent.session_manager import SessionManager


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
