"""
nodes/planning_subagent.py - Planning subagent definition.
Analyzes user request and writes plan.md dynamically per run via a deep agent
backed by deepagents FileSystemBackend and LangChain's TODOListMiddleware.
"""

from typing import Any

import os

from deepagents import create_deep_agent
from deepagents.backends import FileSystemBackend
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.types import interrupt

from agent.harness import HarnessGuard


PLANNING_SYSTEM_PROMPT = """You are the planning-agent for an autonomous coding harness.
Your task is to analyze the user's software request and produce a comprehensive plan.md in the root directory.

Follow these strict requirements:
1. Focus strictly on scope, requirements, architectural constraints, acceptance criteria, and file lists.
2. Do not write full code or dive into lower-level component implementations here.
3. If new dependencies are needed, explicitly flag them in plan.md.
4. Output the plan cleanly into plan.md.
"""


class _WritePlanArgs(BaseModel):
    content: str = Field(description="The full Markdown content to write to plan.md.")


def make_planning_subagent(
    root_dir: str = ".",
    extra_tools: list[Any] | None = None,
) -> Any:
    """
    Constructs a deep planning agent that:
    - Uses ChatOllama (gpt-oss:120b-cloud, temperature=0.2) as the LLM.
    - Uses FileSystemBackend scoped to *root_dir* so it can read files.
    - Exposes a HarnessGuard-protected ``write_plan`` tool for writing plan.md,
      so every plan write is gated by permissions.json (defaults to "ask").
    - Wraps the agent with LangChain's TODOListMiddleware for structured TODO tracking.

    Args:
        root_dir:    Workspace root; the FileSystemBackend is anchored here.
        extra_tools: Optional additional tools to expose to the agent.

    Returns:
        A compiled deep agent ready to be invoked.
    """
    guard = HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    plan_path = os.path.join(root_dir, "plan.md")

    def _write_plan(content: str) -> str:
        """Guard-enforced write of plan.md."""
        guard.enforce(
            "Edit(plan.md)",
            interrupt_fn=interrupt,
            description="Planning agent requests write to plan.md",
        )
        os.makedirs(os.path.dirname(os.path.abspath(plan_path)), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"plan.md written successfully ({len(content)} chars)."

    write_plan_tool = StructuredTool.from_function(
        func=_write_plan,
        name="write_plan",
        description=(
            "Write the complete plan to plan.md. "
            "Use this tool (not a raw file-write) to ensure HarnessGuard approves the write. "
            "Provide the full Markdown content as the 'content' argument."
        ),
        args_schema=_WritePlanArgs,
    )

    llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)
    backend = FileSystemBackend(root_dir=root_dir)
    middleware = [TodoListMiddleware()]

    tools: list[Any] = [write_plan_tool] + list(extra_tools or [])

    return create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        middleware=middleware,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Convenience spec dict kept for harnesses that wire subagents declaratively.
# ---------------------------------------------------------------------------
planning_subagent: dict[str, Any] = {
    "name": "planning-agent",
    "description": "Use FIRST for any new build request. Writes plan.md dynamically.",
    "system_prompt": PLANNING_SYSTEM_PROMPT,
    "factory": make_planning_subagent,
}
