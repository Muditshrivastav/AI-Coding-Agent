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


def test_create_initial_state():
    from agent.state import create_initial_state
    init_state = create_initial_state("Implement feature X")
    assert init_state["user_request"] == "Implement feature X"
    assert init_state["current_stage"] == "planning"
    assert init_state["verify_passed"] is False
    assert init_state["verify_attempts"] == 0
    assert len(init_state["messages"]) == 1
    assert init_state["messages"][0]["content"] == "Implement feature X"
