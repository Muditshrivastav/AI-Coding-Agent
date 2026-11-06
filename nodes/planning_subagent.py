"""
nodes/planning_subagent.py - Planning subagent definition.
Analyzes user request and writes PLAN.md dynamically per run via a deep agent
backed by deepagents FilesystemBackend and MemoryMiddleware.
"""

from typing import Any

import os

from deepagents import create_deep_agent, MemoryMiddleware
from deepagents.backends import FilesystemBackend
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.types import interrupt

from agent.guardrail import HarnessGuard
from agent.task_plan import create_write_task_plan_tool
from frameworks.deepagents_backend import GuardedShellBackend
from frameworks.model_config import PLANNER_MODEL, create_chat_model


PLANNING_SYSTEM_PROMPT = """You are the planning-agent for an autonomous coding harness.
Your task is to analyze the user's software request and produce a comprehensive PLAN.md in the root directory.

Follow these strict requirements:
1. Focus strictly on scope, requirements, architectural constraints, acceptance criteria, and file lists.
2. Do not write full code or dive into lower-level component implementations here.
3. If new dependencies are needed, explicitly flag them in PLAN.md.
4. Read /AGENTS.md and /harness/TASK.md before planning.
5. Write an ordered, dependency-aware task list using write_task_plan. Every task must
   include a unique id, title, acceptance_criteria, and one verify_command that can
   independently prove that task is complete.
6. Output the plan cleanly into PLAN.md.
"""


class _WritePlanArgs(BaseModel):
    content: str = Field(description="The full Markdown content to write to PLAN.md.")


def make_planning_subagent(
    root_dir: str = ".",
    extra_tools: list[Any] | None = None,
    guard: HarnessGuard | None = None,
    model: str | None = None,
) -> Any:
    """
    Constructs a deep planning agent that:
    - Uses the centrally configured planner model with model-level retries.
    - Uses FileSystemBackend scoped to *root_dir* so it can read files.
    - Exposes a HarnessGuard-protected ``write_plan`` tool for writing PLAN.md,
      so every plan write is gated by the configured permission profile.
    - Wraps the agent with LangChain's TODOListMiddleware for structured TODO tracking.

    Args:
        root_dir:    Workspace root; the FileSystemBackend is anchored here.
        extra_tools: Optional additional tools to expose to the agent.

    Returns:
        A compiled deep agent ready to be invoked.
    """
    guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    plan_path = os.path.join(root_dir, "PLAN.md")

    def _write_plan(content: str) -> str:
        """Guard-enforced write of PLAN.md."""
        guard.enforce(
            "Edit(PLAN.md)",
            interrupt_fn=interrupt,
            description="Planning agent requests write to PLAN.md",
        )
        os.makedirs(os.path.dirname(os.path.abspath(plan_path)), exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"PLAN.md written successfully ({len(content)} chars)."

    write_plan_tool = StructuredTool.from_function(
        func=_write_plan,
        name="write_plan",
        description=(
            "Write the complete plan to PLAN.md. "
            "Use this tool (not a raw file-write) to ensure HarnessGuard approves the write. "
            "Provide the full Markdown content as the 'content' argument."
        ),
        args_schema=_WritePlanArgs,
    )

    primary_llm = create_chat_model(model or PLANNER_MODEL, temperature=0.2)
    backend = GuardedShellBackend(FilesystemBackend(root_dir=root_dir), guard)
    middleware = [
        MemoryMiddleware(
            backend=backend,
            sources=["/AGENTS.md", "/harness/TASK.md"],
        )
    ]

    tools: list[Any] = [
        write_plan_tool,
        create_write_task_plan_tool(root_dir, guard),
        *(extra_tools or []),
    ]

    agent_kwargs = {
        "backend": backend,
        "system_prompt": PLANNING_SYSTEM_PROMPT,
        "middleware": middleware,
        "tools": tools,
    }
    return create_deep_agent(model=primary_llm, **agent_kwargs)


# ---------------------------------------------------------------------------
# Convenience spec dict kept for harnesses that wire subagents declaratively.
# ---------------------------------------------------------------------------
planning_subagent: dict[str, Any] = {
    "name": "planning-agent",
    "description": "Use FIRST for any new build request. Writes PLAN.md and harness/tasks.json.",
    "system_prompt": PLANNING_SYSTEM_PROMPT,
}
