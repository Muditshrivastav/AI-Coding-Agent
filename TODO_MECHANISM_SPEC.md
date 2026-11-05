# Autonomous Task Loop

This document describes the task mechanism currently implemented by `agent/repair_loop.py`. The loop uses a structured task plan rather than Deep Agents' self-maintained `write_todos` list.

## Task plan

The planner calls `write_task_plan`, which validates and writes `harness/tasks.json`. The model is defined in `agent/task_plan.py`.

Each task contains:

```json
{
  "id": "unique-lowercase-task-id",
  "title": "Short task title",
  "acceptance_criteria": ["A verifiable condition"],
  "verify_command": "pytest -q tests/test_feature.py",
  "status": "pending",
  "attempts": 0,
  "replanned": false,
  "notes": [],
  "verification_result": null,
  "tool_verification_results": []
}
```

`status` is `pending`, `in_progress`, `done`, `failed`, or `blocked`. Task IDs must be unique. A plan must contain at least one task, and every task must include acceptance criteria and an executable verification command.

## Execution and state

The LangGraph controller follows:

```text
request_start_approval -> plan_tasks -> pick_next_task
                                         |
                 +-----------------------+-------------------+
                 |                       |                   |
             execute_task             final_verify       budget_exceeded
                 |
             verify_task
                 |
             finish_task
          retry / replan / next
```

For each attempt, the controller starts a fresh build-agent invocation with the request, current task, acceptance criteria, attempt number, prior failure feedback, and orientation files (`AGENTS.md`, `TASK.md`, `progress.md`). It does not carry the full previous task conversation into the next task. Tool activity is streamed with subgraphs enabled so structured `run_verification` results are available to the controller.

The loop independently executes `verify_command` using an argv-based subprocess. A task is marked `done` only when this command exits 0 and no captured structured verification result reports failure. Verification results and notes are persisted in `harness/tasks.json` and the graph checkpoint. `harness/progress.md` is regenerated after task state transitions.

## Retries and limits

Defaults are four attempts per task, 32 task attempts per run, 120 wall-clock minutes, and 200,000 reported tokens. Override them with:

- `AGENT_MAX_TASK_ATTEMPTS`
- `AGENT_MAX_ITERATIONS`
- `AGENT_MAX_WALL_CLOCK_MINUTES`
- `AGENT_MAX_TOKENS`

Repeated tool calls with identical names and arguments trigger replanning. A failed task is retried with the real verification output; after another failed attempt, the planner can revise that task. Once the attempt limit is reached, the task is marked `blocked`, and the loop moves to the next task. A run-level budget limit marks unfinished tasks blocked and returns `budget_exceeded`.

## Completion

After all tasks are attempted, the controller reruns verification for tasks marked done. A run is:

- `complete` when every task is still verified done;
- `partial` when one or more tasks are blocked or fail final verification;
- `budget_exceeded` when a run-level limit stops execution.

Human approval interrupts remain checkpointed. API responses surface them as `awaiting_approval`; `resume(thread_id, approved=...)` continues after a decision. A run interrupted by process restart is recoverable only when Postgres checkpointing is configured.

## Verification tools

`run_verification` returns a structured JSON result with `passed`, `tier`, `exit_code`, and `stderr_tail`. The `VerificationToolset` supports per-call timeout overrides and tier defaults: lint 120 seconds, tests 900 seconds, builds 1800 seconds, and E2E 600 seconds. The autonomous task loop does not currently have a browser E2E tier; Chrome DevTools MCP is available only as an optional inspection tool.

## Artifacts and limitations

- `PLAN.md` contains the human-readable plan.
- `harness/tasks.json` contains the structured plan and task outcomes.
- `harness/progress.md` is the current human-readable progress view.
- `harness/TASK.md` is per-request scope; `harness/AGENTS.md` is permanent harness guidance.

The loop is sequential. It does not infer a dependency graph between tasks, and it does not automatically commit, push, or deploy completed tasks.
