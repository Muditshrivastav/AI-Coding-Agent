"""Structured task plans persisted for the autonomous execution loop."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field, model_validator

from agent.guardrail import HarnessGuard


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskItem(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    title: str = Field(min_length=1)
    acceptance_criteria: list[str] = Field(min_length=1)
    verify_command: str = Field(min_length=1)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    replanned: bool = False
    notes: list[str] = Field(default_factory=list)
    verification_result: dict[str, Any] | None = None
    tool_verification_results: list[dict[str, Any]] = Field(default_factory=list)


class TaskPlan(BaseModel):
    tasks: list[TaskItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_ids(self) -> TaskPlan:
        identifiers = [task.id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Task ids must be unique.")
        return self


class WriteTaskPlanArgs(BaseModel):
    tasks: list[TaskItem] = Field(
        min_length=1,
        description="Ordered tasks with acceptance criteria and verification commands.",
    )


def create_write_task_plan_tool(
    root_dir: str,
    guard: HarnessGuard,
) -> StructuredTool:
    path = Path(root_dir) / "harness" / "tasks.json"

    def write_task_plan(tasks: list[TaskItem]) -> str:
        guard.enforce(
            "Write(harness/tasks.json)",
            interrupt_fn=interrupt,
            description="Planning agent requests to persist the structured task plan",
        )
        plan = TaskPlan(tasks=tasks)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return f"Saved {len(plan.tasks)} tasks to harness/tasks.json."

    return StructuredTool.from_function(
        func=write_task_plan,
        name="write_task_plan",
        description=(
            "Persist the complete task list as harness/tasks.json. Each task must "
            "include an acceptance_criteria list and one executable verify_command."
        ),
        args_schema=WriteTaskPlanArgs,
    )


def load_task_plan(path: str | Path) -> TaskPlan:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    return TaskPlan.model_validate(raw)
