import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.repair_loop import _configured_budget, build_repair_graph
from agent.guardrail import HarnessGuard
from agent.task_plan import (
    TaskPlan,
    create_write_task_plan_tool,
    load_task_plan,
)


class FakePlanner:
    def __init__(self, root: Path, tasks: list[dict]):
        self.path = root / "harness" / "tasks.json"
        self.tasks = tasks
        self.calls = []

    async def ainvoke(self, payload, *, config):
        prompt = payload["messages"][-1]["content"]
        self.calls.append(prompt)
        if "Replan only this" in prompt:
            replanned = next(task for task in self.tasks if task["id"] == "blocked")
            tasks = [replanned]
        else:
            tasks = self.tasks
        plan = TaskPlan.model_validate({"tasks": tasks})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(plan.model_dump(mode="json")),
            encoding="utf-8",
        )
        return {"messages": []}


class RetryFakePlanner(FakePlanner):
    async def ainvoke(self, payload, *, config):
        prompt = payload["messages"][-1]["content"]
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return {"messages": []}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        plan = TaskPlan.model_validate({"tasks": self.tasks})
        self.path.write_text(
            json.dumps(plan.model_dump(mode="json")),
            encoding="utf-8",
        )
        return {"messages": []}


class FakeBuildAgent:
    def __init__(self):
        self.prompts = []

    async def astream(
        self,
        payload,
        *,
        config,
        stream_mode,
        subgraphs,
    ):
        self.prompts.append(payload["messages"][-1]["content"])
        message = SimpleNamespace(type="ai", content="Attempt finished.", tool_calls=[])
        yield ("build-agent", {"build-agent": {"messages": [message]}})


def _task(task_id: str) -> dict:
    return {
        "id": task_id,
        "title": f"Implement {task_id}",
        "acceptance_criteria": [f"{task_id} is implemented"],
        "verify_command": f"verify {task_id}",
        "status": "pending",
        "attempts": 0,
        "notes": [],
    }


def test_task_loop_budget_configuration_is_validated(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_TOKENS", "1234")
    assert _configured_budget("AGENT_MAX_TOKENS", 200_000) == 1234

    monkeypatch.setenv("AGENT_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        _configured_budget("AGENT_MAX_TOKENS", 200_000)


def test_write_task_plan_persists_under_strict_workspace_permissions(tmp_path):
    permissions_path = tmp_path / "harness" / "permissions.json"
    permissions_path.parent.mkdir(parents=True)
    permissions_path.write_text(
        json.dumps(
            {
                "allow": ["Write(harness/tasks.json)"],
                "ask": ["Write(*)", "Write(**/*)"],
                "deny": [],
            }
        ),
        encoding="utf-8",
    )
    guard = HarnessGuard(str(permissions_path))
    tool = create_write_task_plan_tool(str(tmp_path), guard)

    result = tool.invoke(
        {
            "tasks": [
                {
                    "id": "test-task",
                    "title": "Persist the plan",
                    "acceptance_criteria": ["Task plan is saved"],
                    "verify_command": "python -m pytest -q",
                }
            ]
        }
    )

    assert result == "Saved 1 tasks to harness/tasks.json."
    assert load_task_plan(tmp_path / "harness" / "tasks.json").tasks[0].id == "test-task"


def test_task_loop_continues_after_blocking_one_task(tmp_path):
    tasks = [_task(task_id) for task_id in ("one", "two", "blocked", "four", "five")]
    planner = FakePlanner(tmp_path, tasks)
    agent = FakeBuildAgent()
    verification_calls = []

    def verify(command, **kwargs):
        verification_calls.append(command)
        failed = command == "verify blocked"
        return {
            "passed": not failed,
            "tier": "task",
            "exit_code": 1 if failed else 0,
            "stderr_tail": "permanent failure" if failed else "",
        }

    graph = build_repair_graph(
        agent,
        planner=planner,
        root_dir=str(tmp_path),
        max_task_attempts=4,
        verify_runner=verify,
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "user_request": "Implement five independent tasks",
                "execution_thread_id": "multi-task-run",
            },
            config={"configurable": {"thread_id": "multi-task-checkpoint"}},
        )
    )

    persisted = load_task_plan(tmp_path / "harness" / "tasks.json")
    statuses = {task.id: task.status.value for task in persisted.tasks}
    assert statuses == {
        "one": "done",
        "two": "done",
        "blocked": "blocked",
        "four": "done",
        "five": "done",
    }
    assert persisted.tasks[2].attempts == 4
    assert len(planner.calls) == 2
    assert result["status"] == "partial"
    assert "Implement blocked" in result["final_output"]
    progress = (tmp_path / "harness" / "progress.md").read_text(encoding="utf-8")
    assert "Implement blocked" in progress
    assert all(
        any(f"Task: Implement {task_id}" in prompt for prompt in agent.prompts)
        for task_id in ("one", "two", "four", "five")
    )
    assert verification_calls.count("verify blocked") == 4


def test_task_loop_uses_structured_verification_results(tmp_path):
    task = _task("structured")
    planner = FakePlanner(tmp_path, [task])

    class ToolResultAgent(FakeBuildAgent):
        async def astream(self, payload, *, config, stream_mode, subgraphs):
            self.prompts.append(payload["messages"][-1]["content"])
            result = {
                "passed": False,
                "tier": "test",
                "exit_code": 1,
                "stderr_tail": "structured pytest failure",
            }
            tool = SimpleNamespace(
                type="tool",
                name="run_verification",
                content=json.dumps(result),
                status="success",
                tool_call_id="verification-call",
            )
            yield ("build-agent", {"build-agent": {"messages": [tool]}})

    graph = build_repair_graph(
        ToolResultAgent(),
        planner=planner,
        root_dir=str(tmp_path),
        max_task_attempts=1,
        verify_runner=lambda command, **kwargs: {
            "passed": True,
            "tier": "task",
            "exit_code": 0,
            "stderr_tail": "",
        },
    )
    result = asyncio.run(
        graph.ainvoke(
            {
                "user_request": "Exercise structured verification",
                "execution_thread_id": "structured-run",
            },
            config={"configurable": {"thread_id": "structured-checkpoint"}},
        )
    )

    task_result = load_task_plan(tmp_path / "harness" / "tasks.json").tasks[0]
    assert task_result.status.value == "blocked"
    assert task_result.tool_verification_results[0]["exit_code"] == 1
    assert any("structured pytest failure" in note for note in task_result.notes)
    assert result["status"] == "partial"


def test_task_loop_retries_if_planner_does_not_persist_task_plan(tmp_path):
    planner = RetryFakePlanner(tmp_path, [_task("retry-plan")])
    graph = build_repair_graph(
        FakeBuildAgent(),
        planner=planner,
        root_dir=str(tmp_path),
        verify_runner=lambda command, **kwargs: {
            "passed": True,
            "tier": "task",
            "exit_code": 0,
            "stderr_tail": "",
        },
    )

    result = asyncio.run(
        graph.ainvoke(
            {
                "user_request": "Implement the retry case",
                "execution_thread_id": "retry-plan-run",
            },
            config={"configurable": {"thread_id": "retry-plan-checkpoint"}},
        )
    )

    assert len(planner.calls) == 2
    assert "did not save harness/tasks.json" in planner.calls[1]
    assert result["status"] == "complete"
