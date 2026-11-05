"""
tests/test_agent_state.py - Unit test verifying AgentState TypedDict schema.
"""

from agent.state import AgentState


def test_agent_state_keys():
    state: AgentState = {
        "user_request": "Build app",
        "retrieved_context": [{"source": "codebase", "content": "test"}],
        "todos": [{"task": "Init", "done": False}],
        "verify_attempts": 0,
        "verify_passed": True,
        "messages": [],
    }
    assert state["verify_passed"] is True
    assert len(state["todos"]) == 1


def test_create_initial_state():
    from agent.state import create_initial_state

    init_state = create_initial_state("Implement feature X")
    assert init_state.get("user_request") == "Implement feature X"
    assert init_state.get("verify_passed") is False
    assert init_state.get("verify_attempts") == 0
    messages = init_state.get("messages")
    assert messages is not None
    assert len(messages) == 1
    assert messages[0]["content"] == "Implement feature X"
