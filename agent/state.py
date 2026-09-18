"""
agent/state.py - Shared state schema for the Coding Agent Harness.
All subagents operate as delegated tasks against this single shared state.
"""

from typing import TypedDict, Literal, Annotated, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    user_request: str
    plan_md: str
    architecture_md: str
    retrieved_context: list[dict[str, Any]]      # Tagged {"source": "codebase" | "external_docs", ...}
    generated_files: list[dict[str, str]]        # Staged [{"path": ..., "content": ...}], pending approval
    approved_files: list[str]                    # Paths that have received approval
    todos: list[dict[str, Any]]                  # Deep Agents' write_todos task state
    verify_attempts: int
    verify_passed: bool
    # Populated by verification_graph hitl_escalation_node when max_attempts is breached.
    # Orchestrators inspect this to surface HITL approval data to the user.
    escalation_payload: dict[str, Any] | None
    messages: Annotated[list[Any], add_messages]
    current_stage: Literal["planning", "design", "build", "verify", "done"]


def create_initial_state(user_request: str) -> AgentState:
    """Creates a fully initialized AgentState dictionary for a new run."""
    return {
        "user_request": user_request,
        "plan_md": "",
        "architecture_md": "",
        "retrieved_context": [],
        "generated_files": [],
        "approved_files": [],
        "todos": [],
        "verify_attempts": 0,
        "verify_passed": False,
        "escalation_payload": None,
        "messages": [{"role": "user", "content": user_request}],
        "current_stage": "planning",
    }
