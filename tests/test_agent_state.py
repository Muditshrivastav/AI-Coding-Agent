"""
tests/test_agent_state.py - Unit test verifying AgentState TypedDict schema.
"""

from agent.state import AgentState


def test_agent_state_keys():
    state: AgentState = {
        "user_request": "Build app",
        "plan_md": "plan content",
        "architecture_md": "arch content",
        "retrieved_context": [{"source": "codebase", "content": "test"}],
        "generated_files": [{"path": "main.py", "content": "print(1)"}],
        "approved_files": ["main.py"],
        "todos": [{"task": "Init", "done": False}],
        "verify_attempts": 0,
        "verify_passed": True,
        "messages": [],
        "current_stage": "planning",
    }
    assert state["current_stage"] == "planning"
    assert state["verify_passed"] is True
    assert len(state["todos"]) == 1
