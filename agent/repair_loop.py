"""Bounded LangGraph orchestration for repairing failed agent tool runs."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Required, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.task_plan import TaskPlan, TaskStatus, load_task_plan

_COMMAND_FAILURE = re.compile(
    r"\[Command failed with exit code [1-9]\d*\]", re.IGNORECASE
)
_DIAGNOSTIC_FAILURE = re.compile(
    r"verification.{0,30}failed|tests? (?:have )?failed|test failure|"
    r"lint.{0,30}failed|build.{0,30}failed|\b\d+ failed\b|"
    r"traceback \(most recent call last\)",
    re.IGNORECASE,
)


class RepairLoopState(TypedDict, total=False):
    user_request: Required[str]
    conversation_history: str
    execution_thread_id: Required[str]
    repair_round: int
    max_repair_rounds: int
    repair_feedback: str
    last_output: str
    final_output: str
    failures: list[str]
    status: str
    start_approved: bool


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _usage_tokens(result: Any) -> int:
    if not isinstance(result, dict):
        return 0
    messages = result.get("messages", [])
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            total += int(usage.get("total_tokens", 0))
    return total


def _tool_failure(status: str, content: str) -> bool:
    if status == "error":
        return True
    return bool(_COMMAND_FAILURE.search(content) or _DIAGNOSTIC_FAILURE.search(content))


def _configured_budget(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        configured = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if configured < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return configured


def build_repair_graph(
    agent: Any,
    *,
    checkpointer: Any = None,
    max_repair_rounds: int = 3,
    guard: Any = None,
    pre_run: Callable[[RepairLoopState], Awaitable[None]] | None = None,
    planner: Any = None,
    root_dir: str = ".",
    max_task_attempts: int | None = None,
    max_iterations: int | None = None,
    max_wall_clock_minutes: int | None = None,
    max_tokens: int | None = None,
    verify_runner: Callable[..., dict[str, Any]] | None = None,
) -> Any:
    """Require run approval, then execute with bounded failure-repair rounds."""
    if planner is not None:
        max_task_attempts = (
            _configured_budget("AGENT_MAX_TASK_ATTEMPTS", 4)
            if max_task_attempts is None
            else max_task_attempts
        )
        max_iterations = (
            _configured_budget("AGENT_MAX_ITERATIONS", 32)
            if max_iterations is None
            else max_iterations
        )
        max_wall_clock_minutes = (
            _configured_budget("AGENT_MAX_WALL_CLOCK_MINUTES", 120)
            if max_wall_clock_minutes is None
            else max_wall_clock_minutes
        )
        max_tokens = (
            _configured_budget("AGENT_MAX_TOKENS", 200_000)
            if max_tokens is None
            else max_tokens
        )
        return _build_autonomous_task_graph(
            agent,
            planner=planner,
            root_dir=root_dir,
            checkpointer=checkpointer,
            guard=guard,
            pre_run=pre_run,
            max_task_attempts=max_task_attempts,
            max_iterations=max_iterations,
            max_wall_clock_minutes=max_wall_clock_minutes,
            max_tokens=max_tokens,
            verify_runner=verify_runner,
        )
    if max_repair_rounds < 0:
        raise ValueError("max_repair_rounds must be non-negative")

    async def request_start_approval(state: RepairLoopState) -> dict[str, Any]:
        if guard is not None:
            try:
                guard.enforce(
                    "AgentRun(start)",
                    interrupt_fn=interrupt,
                    description="Start the coding agent run",
                )
            except PermissionError as exc:
                return {
                    "start_approved": False,
                    "status": "blocked",
                    "final_output": str(exc),
                }

        if pre_run is not None:
            await pre_run(state)
        return {"start_approved": True}

    async def execute_round(
        state: RepairLoopState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        writer = get_stream_writer()
        round_number = state.get("repair_round", 0)
        writer({"type": "iteration_start", "iteration": round_number + 1})

        prompt = state.get("repair_feedback") or state["user_request"]
        history = state.get("conversation_history", "").strip()
        if history and round_number == 0:
            prompt = f"Conversation history:\n{history}\n\nCurrent request:\n{prompt}"

        child_thread_id = state["execution_thread_id"]
        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": child_thread_id,
            },
        }
        latest_tool_results: dict[str, tuple[str, str]] = {}
        output = ""

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for update in chunk.values():
                messages = (
                    update.get("messages", []) if isinstance(update, dict) else []
                )
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    message_type = getattr(message, "type", "")
                    if message_type == "tool":
                        name = getattr(message, "name", "tool") or "tool"
                        content = _message_text(getattr(message, "content", ""))
                        latest_tool_results[name] = (
                            content,
                            getattr(message, "status", ""),
                        )
                        writer(
                            {
                                "type": "tool_result",
                                "tool": name,
                                "output": content.splitlines()[0][:120]
                                if content
                                else "",
                                "id": getattr(message, "tool_call_id", "") or "",
                                "lines": content.count("\n") + 1 if content else 0,
                            }
                        )
                    elif message_type == "ai":
                        tool_calls = getattr(message, "tool_calls", []) or []
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                name = tool_call.get("name", "tool")
                                arguments = tool_call.get("args", {})
                                call_id = tool_call.get("id", "")
                            else:
                                name = getattr(tool_call, "name", "tool")
                                arguments = getattr(tool_call, "args", {})
                                call_id = getattr(tool_call, "id", "")
                            writer(
                                {
                                    "type": "tool_call",
                                    "tool": name,
                                    "input": json.dumps(arguments, default=str)[:120],
                                    "id": call_id,
                                }
                            )
                        content = _message_text(getattr(message, "content", ""))
                        if content.strip() and not tool_calls:
                            output = content.strip()
                            writer({"type": "text_delta", "text": content})

        failures = [
            f"{name}: {content[:2000]}"
            for name, (content, status) in latest_tool_results.items()
            if _tool_failure(status, content)
        ]
        writer(
            {
                "type": "iteration_complete",
                "iteration": round_number + 1,
                "failed": bool(failures),
            }
        )

        result: dict[str, Any] = {
            "last_output": output,
            "failures": failures,
            "final_output": output or "Task completed.",
        }
        if failures and round_number >= state.get(
            "max_repair_rounds", max_repair_rounds
        ):
            details = "\n".join(f"- {failure}" for failure in failures)
            result.update(
                {
                    "status": "needs_attention",
                    "final_output": (f"{output}\n\n" if output else "")
                    + (
                        "⚠️ Automatic repair limit reached before the failing checks passed.\n\n"
                        f"Latest failure details:\n{details}"
                    ),
                }
            )
        elif not failures:
            result["status"] = "complete"
        return result

    def prepare_repair(state: RepairLoopState) -> dict[str, Any]:
        next_round = state.get("repair_round", 0) + 1
        details = "\n".join(f"- {failure}" for failure in state.get("failures", []))
        feedback = (
            "The previous work round reported failures. Continue working in the same "
            "workspace, diagnose and fix these exact issues, then rerun the relevant "
            "verification. Do not merely explain the failures.\n\n"
            f"Failure details:\n{details}\n\n"
            f"Previous response:\n{state.get('last_output', '')}"
        )
        get_stream_writer()(
            {
                "type": "repair_started",
                "iteration": next_round + 1,
                "failures": state.get("failures", []),
            }
        )
        return {"repair_round": next_round, "repair_feedback": feedback}

    def route_after_round(state: RepairLoopState) -> str:
        if not state.get("failures"):
            return END
        if state.get("repair_round", 0) >= state.get(
            "max_repair_rounds", max_repair_rounds
        ):
            return END
        return "prepare_repair"

    def route_after_start(state: RepairLoopState) -> str:
        if not state.get("start_approved", False):
            return END
        return "execute_round"

    graph = StateGraph(RepairLoopState)
    graph.add_node("request_start_approval", request_start_approval)
    graph.add_node("execute_round", execute_round)
    graph.add_node("prepare_repair", prepare_repair)
    graph.add_edge(START, "request_start_approval")
    graph.add_conditional_edges("request_start_approval", route_after_start)
    graph.add_conditional_edges("execute_round", route_after_round)
    graph.add_edge("prepare_repair", "execute_round")
    return graph.compile(checkpointer=checkpointer)


class TaskLoopState(TypedDict, total=False):
    user_request: Required[str]
    execution_thread_id: Required[str]
    tasks: list[dict[str, Any]]
    current_task: dict[str, Any]
    task_index: int
    max_task_attempts: int
    max_iterations: int
    max_wall_clock_minutes: int
    max_tokens: int
    iteration_count: int
    token_count: int
    started_at: float
    last_output: str
    task_feedback: str
    verification_result: dict[str, Any]
    tool_verification_results: list[dict[str, Any]]
    status: str
    final_output: str
    next_action: str
    last_failure: str


def _state_current_task(state: TaskLoopState) -> dict[str, Any]:
    task = state.get("current_task")
    if not isinstance(task, dict):
        raise TypeError("Task loop state has no current task.")
    return task


def _state_tasks(state: TaskLoopState) -> list[dict[str, Any]]:
    tasks = state.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError("Task loop state has no task list.")
    return tasks


def _state_task_index(state: TaskLoopState) -> int:
    index = state.get("task_index")
    if not isinstance(index, int):
        raise TypeError("Task loop state has no current task index.")
    return index


def _state_verification_result(state: TaskLoopState) -> dict[str, Any]:
    result = state.get("verification_result")
    if not isinstance(result, dict):
        raise TypeError("Task loop state has no verification result.")
    return result


def _verification_payload(message: Any) -> dict[str, Any] | None:
    content = getattr(message, "content", None)
    if isinstance(content, dict) and "passed" in content:
        return content
    if isinstance(content, str):
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return None
        return result if isinstance(result, dict) and "passed" in result else None
    return None


def _execute_verification_command(
    command: str,
    *,
    root_dir: str,
    timeout_seconds: int,
    guard: Any = None,
) -> dict[str, Any]:
    if guard is not None:
        guard.enforce(
            f"Bash({command})",
            interrupt_fn=interrupt,
            description=f"Verify task with command: {command}",
        )
    arguments = shlex.split(command, posix=os.name != "nt")
    if not arguments:
        raise ValueError("A task verify_command cannot be empty.")
    try:
        result = subprocess.run(
            arguments,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        output = "\n".join(part for part in (stdout, stderr) if part)
        return {
            "passed": False,
            "tier": "task",
            "exit_code": 124,
            "stderr_tail": f"Verification timed out after {timeout_seconds}s.\n{output}"[
                -4000:
            ],
        }
    except FileNotFoundError as exc:
        return {
            "passed": False,
            "tier": "task",
            "exit_code": 127,
            "stderr_tail": str(exc),
        }
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return {
        "passed": result.returncode == 0,
        "tier": "task",
        "exit_code": result.returncode,
        "stderr_tail": output[-4000:],
    }


def _build_autonomous_task_graph(
    agent: Any,
    *,
    planner: Any,
    root_dir: str,
    checkpointer: Any,
    guard: Any,
    pre_run: Callable[[RepairLoopState], Awaitable[None]] | None,
    max_task_attempts: int,
    max_iterations: int,
    max_wall_clock_minutes: int,
    max_tokens: int,
    verify_runner: Callable[..., dict[str, Any]] | None,
) -> Any:
    if max_task_attempts < 1:
        raise ValueError("max_task_attempts must be at least one.")
    if min(max_iterations, max_wall_clock_minutes, max_tokens) < 1:
        raise ValueError("Task-loop budgets must be positive.")

    root = Path(root_dir).resolve()
    tasks_path = root / "harness" / "tasks.json"
    progress_path = root / "harness" / "progress.md"
    run_verification = verify_runner or _execute_verification_command

    def persist_tasks(tasks: list[dict[str, Any]]) -> None:
        plan = TaskPlan.model_validate({"tasks": tasks})
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.write_text(
            json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )

    def update_progress(
        tasks: list[dict[str, Any]],
        *,
        stage: str,
        current_task: dict[str, Any] | None = None,
    ) -> None:
        done = [item["title"] for item in tasks if item["status"] == TaskStatus.DONE]
        pending = [
            item["title"]
            for item in tasks
            if item["status"] in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}
        ]
        blocked = [
            item["title"] for item in tasks if item["status"] == TaskStatus.BLOCKED
        ]
        current = current_task["title"] if current_task else "none"
        progress = (
            "# Build Progress\n\n"
            f"## Current Stage\n`{stage}`\n\n"
            f"## Current Task\n{current}\n\n"
            f"## Completed Tasks\n{chr(10).join(f'- {title}' for title in done) or 'None.'}\n\n"
            f"## Pending Tasks\n{chr(10).join(f'- {title}' for title in pending) or 'None.'}\n\n"
            f"## Blocked Tasks\n{chr(10).join(f'- {title}' for title in blocked) or 'None.'}\n"
        )
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(progress, encoding="utf-8")

    def budget_exceeded(state: TaskLoopState) -> bool:
        elapsed = time.time() - state.get("started_at", time.time())
        return (
            state.get("iteration_count", 0)
            >= state.get("max_iterations", max_iterations)
            or elapsed
            >= state.get("max_wall_clock_minutes", max_wall_clock_minutes) * 60
            or state.get("token_count", 0) >= state.get("max_tokens", max_tokens)
        )

    async def request_start_approval(state: TaskLoopState) -> dict[str, Any]:
        if guard is not None:
            try:
                guard.enforce(
                    "AgentRun(start)",
                    interrupt_fn=interrupt,
                    description="Start the autonomous coding agent run",
                )
            except PermissionError as exc:
                return {"status": "blocked", "final_output": str(exc)}
        if pre_run is not None:
            await pre_run(state)  # type: ignore[arg-type]
        return {
            "status": "running",
            "started_at": time.time(),
            "max_task_attempts": max_task_attempts,
            "max_iterations": max_iterations,
            "max_wall_clock_minutes": max_wall_clock_minutes,
            "max_tokens": max_tokens,
            "iteration_count": 0,
            "token_count": 0,
        }

    async def plan_tasks(
        state: TaskLoopState, config: RunnableConfig
    ) -> dict[str, Any]:
        supplied_tasks = state.get("tasks")
        planner_tokens = 0
        if supplied_tasks:
            plan = TaskPlan.model_validate({"tasks": supplied_tasks})
        else:
            if tasks_path.exists():
                tasks_path.unlink()
            planner_thread_id = f"{state['execution_thread_id']}:planner"
            planner_config = {
                **config,
                "configurable": {
                    **config.get("configurable", {}),
                    "thread_id": planner_thread_id,
                },
            }
            prompt = (
                "Plan the current request only. Read /AGENTS.md and /harness/TASK.md. "
                "Write PLAN.md, then call write_task_plan with a complete ordered list "
                "of independent tasks. Each task needs a unique id, acceptance criteria, "
                "and an executable verify_command. Do not implement any tasks yet.\n\n"
                f"Request:\n{state['user_request']}"
            )
            planner_result = await planner.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=planner_config,
            )
            planner_tokens = _usage_tokens(planner_result)
            if not tasks_path.is_file():
                raise RuntimeError(
                    "Planning agent did not persist a structured task list to harness/tasks.json."
                )
            plan = load_task_plan(tasks_path)
        tasks = [item.model_dump(mode="json") for item in plan.tasks]
        persist_tasks(tasks)
        update_progress(tasks, stage="planned")
        return {
            "tasks": tasks,
            "task_index": -1,
            "token_count": state.get("token_count", 0) + planner_tokens,
        }

    def pick_next_task(state: TaskLoopState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        if budget_exceeded(state):
            for item in tasks:
                if item["status"] in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}:
                    item["status"] = TaskStatus.BLOCKED
                    item["notes"].append("Run-level execution budget exceeded.")
            persist_tasks(tasks)
            update_progress(tasks, stage="budget_exceeded")
            return {
                "tasks": tasks,
                "status": "budget_exceeded",
                "final_output": "Run-level iteration, time, or token budget exceeded.",
                "next_action": "end",
            }
        for index, item in enumerate(tasks):
            if item["status"] in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}:
                current = dict(item)
                item["status"] = TaskStatus.IN_PROGRESS
                persist_tasks(tasks)
                update_progress(tasks, stage="build", current_task=current)
                return {
                    "tasks": tasks,
                    "current_task": current,
                    "task_index": index,
                    "next_action": "execute",
                }
        return {"next_action": "final_verify"}

    async def execute_task(
        state: TaskLoopState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        task = _state_current_task(state)
        attempt = int(task.get("attempts", 0)) + 1
        task["attempts"] = attempt
        task["status"] = TaskStatus.IN_PROGRESS
        tasks = _state_tasks(state)
        tasks[_state_task_index(state)] = task
        persist_tasks(tasks)
        update_progress(tasks, stage="build", current_task=task)

        writer = get_stream_writer()
        writer(
            {
                "type": "task_start",
                "task_id": task["id"],
                "attempt": attempt,
                "title": task["title"],
            }
        )
        criteria = "\n".join(f"- {item}" for item in task["acceptance_criteria"])
        prompt = (
            "Implement only the current task. Do not continue to other tasks or carry "
            "over earlier conversation. Read /AGENTS.md, /harness/TASK.md, and "
            "/harness/progress.md, then make the smallest complete change. Run the "
            "task's verify_command and report its raw result; the orchestrator will "
            "run the command independently before marking this task done. Never edit "
            "tests merely to make them pass.\n\n"
            f"Task: {task['title']}\n"
            f"Acceptance criteria:\n{criteria}\n"
            f"Required verification command: {task['verify_command']}\n"
            f"Attempt: {attempt}/{state.get('max_task_attempts', max_task_attempts)}\n"
            f"Previous failure feedback:\n{state.get('task_feedback', 'None.')}"
        )
        child_thread_id = (
            f"{state['execution_thread_id']}:task:{task['id']}:attempt:{attempt}"
        )
        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": child_thread_id,
            },
        }
        output = ""
        tool_results: list[dict[str, Any]] = []
        latest_structured_verification = None
        last_call: str | None = None
        repeated_calls = 0
        stuck = False
        token_count = state.get("token_count", 0)

        async for event in agent.astream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
            stream_mode="updates",
            subgraphs=True,
        ):
            chunk = event[1] if isinstance(event, tuple) and len(event) == 2 else event
            if not isinstance(chunk, dict):
                continue
            for update in chunk.values():
                messages = (
                    update.get("messages", []) if isinstance(update, dict) else []
                )
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    message_type = getattr(message, "type", "")
                    if message_type == "tool":
                        name = getattr(message, "name", "tool") or "tool"
                        content = _message_text(getattr(message, "content", ""))
                        result = {
                            "name": name,
                            "content": content,
                            "status": getattr(message, "status", ""),
                            "verification": (
                                _verification_payload(message)
                                if name == "run_verification"
                                else None
                            ),
                        }
                        tool_results.append(result)
                        if name == "run_verification":
                            latest_structured_verification = result["verification"]
                        writer(
                            {
                                "type": "tool_result",
                                "tool": name,
                                "output": content.splitlines()[0][:120]
                                if content
                                else "",
                                "id": getattr(message, "tool_call_id", "") or "",
                                "lines": content.count("\n") + 1 if content else 0,
                            }
                        )
                    elif message_type == "ai":
                        usage = getattr(message, "usage_metadata", None) or {}
                        token_count += int(usage.get("total_tokens", 0))
                        calls = getattr(message, "tool_calls", []) or []
                        for call in calls:
                            signature = json.dumps(
                                {
                                    "name": call.get("name", "tool"),
                                    "args": call.get("args", {}),
                                }
                                if isinstance(call, dict)
                                else {
                                    "name": getattr(call, "name", "tool"),
                                    "args": getattr(call, "args", {}),
                                },
                                sort_keys=True,
                                default=str,
                            )
                            repeated_calls = (
                                repeated_calls + 1 if signature == last_call else 1
                            )
                            last_call = signature
                            if repeated_calls >= 3:
                                stuck = True
                            writer(
                                {
                                    "type": "tool_call",
                                    "tool": json.loads(signature)["name"],
                                    "input": signature[:120],
                                }
                            )
                        content = _message_text(getattr(message, "content", ""))
                        if content.strip() and not calls:
                            output = content.strip()
                            writer({"type": "text_delta", "text": content})

        return {
            "last_output": output,
            "tool_results": tool_results,
            "structured_verification": latest_structured_verification,
            "tool_verification_results": [
                item["verification"]
                for item in tool_results
                if item.get("verification") is not None
            ],
            "stuck_detected": stuck,
            "token_count": token_count,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def verify_task(state: TaskLoopState) -> dict[str, Any]:
        task = _state_current_task(state)
        if budget_exceeded(state):
            result = {
                "passed": False,
                "tier": "task",
                "exit_code": 124,
                "stderr_tail": "Run-level execution budget exceeded before task verification.",
            }
        else:
            result = run_verification(
                task["verify_command"],
                root_dir=str(root),
                timeout_seconds=1800,
                guard=guard,
            )
        if "passed" not in result or "exit_code" not in result:
            raise TypeError("verify_runner must return passed and exit_code fields.")
        return {"verification_result": result}

    def finish_task(state: TaskLoopState) -> dict[str, Any]:
        task = _state_current_task(state)
        result = _state_verification_result(state)
        tasks = _state_tasks(state)
        index = _state_task_index(state)
        task["verification_result"] = result
        task["tool_verification_results"] = state.get("tool_verification_results", [])
        tool_failures = [
            item
            for item in task["tool_verification_results"]
            if not item.get("passed") or item.get("exit_code") != 0
        ]
        if result["passed"] and result["exit_code"] == 0 and not tool_failures:
            task["status"] = TaskStatus.DONE
            task["notes"].append(
                f"Verified successfully on attempt {task['attempts']}."
            )
            action = "next"
            feedback = ""
        else:
            error = str(
                result.get("stderr_tail")
                or (
                    tool_failures[-1].get("stderr_tail")
                    if tool_failures
                    else f"exit code {result['exit_code']}"
                )
            )
            task["status"] = TaskStatus.PENDING
            task["notes"].append(f"Attempt {task['attempts']} failed: {error[-1000:]}")
            attempt = int(task["attempts"])
            if attempt >= state.get("max_task_attempts", max_task_attempts):
                task["status"] = TaskStatus.BLOCKED
                task["notes"].append("Maximum task attempts reached.")
                action = "next"
                feedback = ""
            elif attempt >= 2 and not task.get("replanned", False):
                action = "replan"
                feedback = (
                    "The task verification failed after a real retry. Replan this task "
                    "with the failure output below, then return one revised task using "
                    "write_task_plan.\n\n"
                    f"Failure output:\n{error}"
                )
            else:
                action = "retry"
                feedback = (
                    "The task verification failed. Fix the issue and retry it; use this "
                    "exact verification output rather than guessing.\n\n"
                    f"Failure output:\n{error}"
                )
            task["notes"].append(error[-1000:])
        tasks[index] = task
        persist_tasks(tasks)
        update_progress(
            tasks,
            stage="verify" if action == "next" else action,
            current_task=task,
        )
        if state.get("stuck_detected") and action not in {"next", "replan"}:
            action = "replan"
            feedback = (
                "Repeated identical tool calls indicate a stuck task. Replan it with "
                "the verification output and do not repeat the same call unchanged."
            )
        if budget_exceeded(state):
            action = "budget"
        return {
            "tasks": tasks,
            "task_feedback": feedback,
            "last_failure": str(result.get("stderr_tail", "")),
            "next_action": action,
        }

    async def replan_task(
        state: TaskLoopState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        task = _state_current_task(state)
        planner_thread_id = (
            f"{state['execution_thread_id']}:replan:{task['id']}:{task['attempts']}"
        )
        planner_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": planner_thread_id,
            },
        }
        planner_result = await planner.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Replan only this blocked-progress task. Read "
                            "/AGENTS.md and /harness/TASK.md, then call write_task_plan "
                            "with a one-task revised plan using the same task id. Do not "
                            "implement the task.\n\n"
                            f"Task:\n{json.dumps(task, indent=2)}\n\n"
                            f"Failure:\n{state.get('last_failure', '')}"
                        ),
                    }
                ]
            },
            config=planner_config,
        )
        if not tasks_path.is_file():
            raise RuntimeError("Planner did not persist its revised task plan.")
        replanned = load_task_plan(tasks_path).tasks[0].model_dump(mode="json")
        if replanned["id"] != task["id"]:
            raise ValueError("A replanned task must retain its original id.")
        replanned["attempts"] = task["attempts"]
        replanned["replanned"] = True
        replanned["notes"] = [
            *task["notes"],
            "Task replanned after repeated verification failure.",
        ]
        tasks = _state_tasks(state)
        tasks[_state_task_index(state)] = replanned
        persist_tasks(tasks)
        update_progress(tasks, stage="replan", current_task=replanned)
        return {
            "tasks": tasks,
            "current_task": replanned,
            "task_feedback": state.get("task_feedback", ""),
            "next_action": "retry",
            "token_count": state.get("token_count", 0) + _usage_tokens(planner_result),
        }

    def final_verify(state: TaskLoopState) -> dict[str, Any]:
        tasks = _state_tasks(state)
        final_failures: list[str] = []
        for task in tasks:
            if task["status"] != TaskStatus.DONE:
                continue
            if budget_exceeded(state):
                task["status"] = TaskStatus.BLOCKED
                task["notes"].append(
                    "Run-level execution budget exceeded during final verification."
                )
                continue
            result = run_verification(
                task["verify_command"],
                root_dir=str(root),
                timeout_seconds=1800,
                guard=guard,
            )
            task["verification_result"] = result
            if not result.get("passed") or result.get("exit_code") != 0:
                task["status"] = TaskStatus.BLOCKED
                reason = str(result.get("stderr_tail") or "Final verification failed.")
                task["notes"].append(f"Final verification failed: {reason[-1000:]}")
                final_failures.append(task["title"])
        persist_tasks(tasks)
        blocked = [
            task["title"] for task in tasks if task["status"] == TaskStatus.BLOCKED
        ]
        complete = all(task["status"] == TaskStatus.DONE for task in tasks)
        status = "complete" if complete else "partial"
        update_progress(tasks, stage="done" if complete else "partial")
        details = "\nBlocked tasks: " + ", ".join(blocked) if blocked else ""
        if final_failures:
            details += "\nFinal verification failed for: " + ", ".join(final_failures)
        return {
            "tasks": tasks,
            "status": status,
            "final_output": (
                "All planned tasks passed their verification commands."
                if complete
                else f"Run finished with blocked tasks.{details}"
            ),
        }

    def route_after_approval(state: TaskLoopState) -> str:
        return "plan_tasks" if state.get("status") == "running" else END

    def route_after_pick(state: TaskLoopState) -> str:
        action = state.get("next_action")
        if action == "execute":
            return "execute_task"
        if action == "end":
            return END
        return "final_verify"

    def route_after_finish(state: TaskLoopState) -> str:
        action = state.get("next_action")
        if action == "retry":
            return "execute_task"
        if action == "replan":
            return "replan_task"
        if action == "budget":
            return "pick_next_task"
        return "pick_next_task"

    graph = StateGraph(TaskLoopState)
    graph.add_node("request_start_approval", request_start_approval)
    graph.add_node("plan_tasks", plan_tasks)
    graph.add_node("pick_next_task", pick_next_task)
    graph.add_node("execute_task", execute_task)
    graph.add_node("verify_task", verify_task)
    graph.add_node("finish_task", finish_task)
    graph.add_node("replan_task", replan_task)
    graph.add_node("final_verify", final_verify)
    graph.add_edge(START, "request_start_approval")
    graph.add_conditional_edges("request_start_approval", route_after_approval)
    graph.add_edge("plan_tasks", "pick_next_task")
    graph.add_conditional_edges("pick_next_task", route_after_pick)
    graph.add_edge("execute_task", "verify_task")
    graph.add_edge("verify_task", "finish_task")
    graph.add_conditional_edges("finish_task", route_after_finish)
    graph.add_edge("replan_task", "execute_task")
    graph.add_edge("final_verify", END)
    return graph.compile(checkpointer=checkpointer)
