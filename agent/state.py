"""
agent/state.py - Shared state schema for the Coding Agent Harness.
All subagents operate as delegated tasks against this single shared state.
"""

from typing import TypedDict, Literal, Annotated, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    user_request: str
    plan_md: str
    architecture_md: str
    retrieved_context: list[dict[str, Any]]      # Tagged {"source": "codebase" | "external_docs", ...}
    generated_files: list[dict[str, str]]        # Staged [{"path": ..., "content": ...}], pending approval
    approved_files: list[str]                    # Paths that have received approval
    todos: list[dict[str, Any]]                  # Deep Agents' write_todos task state
    verify_attempts: int
    verify_passed: bool
    messages: Annotated[list[Any], add_messages]
    current_stage: Literal["planning", "design", "build", "verify", "done"]
