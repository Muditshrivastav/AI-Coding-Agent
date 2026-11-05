# AI-Coding-Agent: Fix List (Major + Minor)

**Goal:** make the agent run long tasks on its own (plan -> build -> verify -> repeat until done) instead of needing one instruction at a time.

**Repo facts these fixes are based on** (verified by reading the code; Deep Agents 0.7.19 per `uv.lock`):
- Nothing in code checks "is the whole task finished?". `agent/repair_loop.py` only re-runs when it finds failure text in tool output.
- `harness/permissions.json` asks for approval on every shell command, file write, and verification run.
- `harness/AGENTS.md` is overwritten on every request and currently holds a stale one-off task.
- `factory` keys in subagent dicts are ignored by Deep Agents, and MCP tools are never loaded.

## Rules for the AI making these changes

1. Work one fix at a time, in the order listed. One commit per fix.
2. Never edit, weaken, or delete a test just to make it pass. Add new tests for each fix.
3. Keep the repo conventions: OOP style, one framework per file under `frameworks/`, no new dependency without adding it to `pyproject.toml` and noting it in the commit message.
4. After each fix run `ruff check .`, `ruff format --check .` and `pytest -q`. Do not proceed if they fail.
5. Keep public method signatures of `CodingAgentHarness.run()` / `resume()` unchanged unless the fix says otherwise.
6. Tests must use `tmp_path` for `permissions.json`, `failures.md`, `harness/` files. Never write to the real `harness/` directory.

---

## MAJOR FIXES

### M1. Stop asking approval for everything
**Problem:** `permissions.json` has `Bash(*)`, `Edit(*)`, `Edit(**/*)`, `Write(*)`, `Write(**/*)`, `Delete(*)`, `run_verification`, `AgentRun(start)` under `ask`. `check_permission` in `agent/guardrail.py` evaluates deny, then ask, then allow, and returns `ask` when nothing matches. `run_shell` in `nodes/build_subagent.py` (around lines 104-117) re-checks `Bash(<cmd>)` even though `execute_command` is in `allow`. So every command and every file write pauses the run.

**Change:**
- Move normal work to `allow`: `Bash(*)`, `Edit(*)`, `Edit(**/*)`, `Write(*)`, `Write(**/*)`, `run_verification`, `AgentRun(start)`, and the GraphRAG tools.
- Keep `deny` exactly as is.
- Keep `ask` only for irreversible actions: `Bash(git push*)`, `Bash(git reset --hard*)`, `Bash(git clean*)`, `Delete(*)`, `Delete(**/*)`, `vercel_deploy`, `render_deploy`, `deploy_to_render`, `mcp__github__create_pull_request`, `mcp__github__merge_pull_request`, `mcp__github__delete_branch`, `mcp__github__push_files`.
- Because `ask` is checked before `allow`, make sure no broad `ask` pattern still shadows the new `allow` patterns.
- Add an optional per-thread approval cache in `HarnessGuard.enforce`: if the user approves with `{"approved": true, "remember": true}`, approve the same signature without asking again for that `thread_id`.
- Add a setting `PERMISSION_PROFILE` (`strict` | `autonomous`, default `strict`) so the current behavior stays available. The `autonomous` profile is the new allow/ask split above. Load it in `HarnessGuard`.
- `autonomous` must only be allowed when `SANDBOX_MODE` is `docker` or `auto` (see m5). With `local` it must log a warning and fall back to `strict`.

**Done when:** a run in `autonomous` mode can write files, run shell commands and run verification with zero interrupts, but still pauses on `git push`, deploys, and deletes. Tests cover each pattern group.

### M2. Make AGENTS.md permanent; put per-request scope somewhere else
**Problem:** `frameworks/agents_md_writer.py` `_write_agents_md` opens `harness/AGENTS.md` with mode `"w"` on every request (called from `prepare_agent_run` in `agent/core.py`). It also feeds the old file back to the LLM as "base conventions". The committed file currently says "All modifications confined to `/nodes/build_subagent.py`" and "Editing any other file is prohibited", which blocks any new, larger task. It also appends unrelated skills (oauth, docker, database, rest-api).

**Change:**
- Rewrite `harness/AGENTS.md` as a permanent, task-independent file: harness rules, verification expectations, never edit tests to pass, never push or deploy without approval, repo conventions, how to use `PLAN.md`, `progress.md`, `failures.md`.
- Change the writer to produce `harness/TASK.md` (not AGENTS.md). Do not read the previous `TASK.md` as context.
- Only inject skills into `TASK.md`, only those with a relevance score above a threshold, max 3, and only their short summaries.
- Make the planning, design and build agents read both `/AGENTS.md` and `/harness/TASK.md`. Update the `MemoryMiddleware(sources=...)` in `nodes/planning_subagent.py` and the orientation step in `BUILD_SYSTEM_PROMPT`.
- Delete the stale generated content from the committed file.

**Done when:** running two different requests back to back never changes `AGENTS.md`, and `TASK.md` only describes the current request.

### M3. Add a real completion loop (the main reason it does not run on its own)
**Problem:** `execute_round` in `agent/repair_loop.py` marks the run `complete` whenever there are no failure strings, even if work is unfinished. `AgentState.todos`, `verify_passed`, `current_stage` are never read by the loop. Retries stop at 3 rounds.

**Change:**
- Have the planning stage output a structured task list (Pydantic model), saved to `harness/tasks.json` and into graph state: `[{id, title, acceptance_criteria, verify_command, status: pending|in_progress|done|failed|blocked, attempts, notes}]`.
- Replace the single `execute_round` with an outer LangGraph loop: `plan -> pick_next_task -> execute_task -> verify_task -> (mark_done | retry | replan | mark_blocked) -> pick_next_task ... -> final_verify -> END`.
- A task is only `done` when its own `verify_command` exits 0. The model saying "done" does not count.
- Per task: up to N attempts (default 4). On repeated failure do not stop: first retry with the real error output, then ask the planner to replan that task, then mark it `blocked` with a reason and continue to the next independent task.
- Run-level limits (configurable): max iterations, max wall-clock minutes, max tokens. Add stuck detection (same tool call with same arguments 3 times in a row, or same failure text twice) that triggers a replan.
- Give each task a fresh agent invocation with only: `AGENTS.md`, `TASK.md`, the task, and `progress.md`. Do not carry the whole conversation.
- Update `harness/progress.md` after every task (current stage, done, pending, blocked).
- Final status is one of `complete`, `partial` (some tasks blocked, list them), `needs_approval`, `budget_exceeded`. Never report `complete` while any task is not `done`.
- Human interrupts still work exactly as now (`__interrupt__`, `resume`).

**Done when:** a 5-task request runs end to end without any prompt from the user, and a test with one permanently failing task ends as `partial` with the other four `done`.

### M4. Register subagents the way Deep Agents actually expects
**Problem:** Deep Agents 0.7.19 `SubAgent` has `name`, `description`, `system_prompt`, `tools`, `model`, `middleware`, `interrupt_on`, `skills`, `permissions`, `response_format`. It has no `factory`. So in `nodes/planning_subagent.py`, `nodes/design_subagent.py`, `nodes/build_subagent.py` the `factory`, `external_tools`, `mcp_clients` keys are ignored. The planner and designer end up as generic subagents without `write_plan`, `write_architecture`, draw.io tools, or `MemoryMiddleware`.

**Change:**
- Build each subagent as either a valid `SubAgent` dict (with `tools=[...]` and `middleware=[...]` set directly) or a `CompiledSubAgent` with a `runnable`.
- Remove the dead keys and the unused `make_*_subagent` / `get_build_dev_tools` paths, or make them the real path (see M5). Do not keep two parallel code paths.
- Use one filename everywhere: `PLAN.md` (see m1).
- Add a test that asserts each subagent has the tools it needs (`write_plan` on planner, `write_architecture` on designer, `execute_command` + `run_verification` on builder).

**Done when:** inspecting the compiled agent shows the planner can call `write_plan` and the designer can call `write_architecture`.

### M5. Actually load the MCP tools (GitHub, Chrome DevTools, Render, draw.io)
**Problem:** `ExternalToolsManager.get_tools(include_mcp=True)` is only called inside `get_build_dev_tools()`, which is only called by `make_build_subagent()`, which nothing calls. So the running agent has no GitHub MCP, Chrome DevTools MCP, Render MCP or draw.io tools. `permissions.json` rules for them are never exercised, and the README's browser E2E check does not exist: `tools/verification_tools.py` has no browser step.

**Change:**
- Add an async constructor, e.g. `await CodingAgentHarness.create(...)`, that loads MCP tools once at startup (with timeouts and per-server failure isolation), wraps each through `HarnessGuard`, and passes them into the subagents.
- Call it from the FastAPI lifespan in `interfaces/api_server.py` and from `cli.py` / `interfaces/acp_server.py`. Close MCP sessions on shutdown.
- If an MCP server is unavailable, log it and continue without it. Never silently pretend it loaded: expose the loaded tool names in a `/health` or `/tools` endpoint.
- Add an E2E verification tier that uses Chrome DevTools MCP (or Playwright if MCP is not available) for frontend tasks. Make it a per-task `verify_command` option in M3. If it is not implemented, remove the E2E claims from `README.md`.

**Done when:** `/tools` lists the GitHub and Chrome DevTools tool names, and a frontend task fails verification when the page throws a console error.

### M6. Durable state and non-blocking runs
**Problem:** `frameworks/langgraph_runtime.py` uses `MemorySaver`, so a restart loses all run state. Supabase stores only session metadata. `agent/core.py` `resume()` uses `self._execution_graphs.get(thread_id, self._agent)`; after a restart that falls back to the wrong graph. `/runs` in `interfaces/api_server.py` (around lines 249-255) awaits the entire run inside the HTTP request.

**Change:**
- Use a Postgres checkpointer (`langgraph-checkpoint-postgres`, `AsyncPostgresSaver`) on the existing Supabase DSN. Keep `MemorySaver` only for tests or when no DSN is set.
- Make `POST /runs` start the run in a background worker and return `{thread_id, status: "running"}` immediately. Add `GET /runs/{thread_id}` (status, current task, progress, pending approval) and keep the SSE stream.
- Make `resume()` rebuild the correct execution graph from the `thread_id` instead of relying on the in-memory dict.
- On startup, mark runs that were `running` before the crash as `interrupted` and allow resuming them.

**Done when:** killing and restarting the API during a run, then calling resume, continues from the last completed task.

### M7. Make failures visible to the loop
**Problem:** the loop streams with `stream_mode="updates"` and no `subgraphs=True`, so tool results inside build-agent are probably invisible; the orchestrator only sees the `task` tool summary. Failure detection also relies on regex over text and keeps only the last result per tool name. (Verify this first with a small test run and log what chunks arrive.)

**Change:**
- Make `run_verification` return a structured result (`passed`, `tier`, `exit_code`, `stderr_tail`) and write it to `harness/tasks.json` / state. The M3 loop reads that, not regexes.
- If needed, stream with `subgraphs=True` so subagent tool calls are visible in the UI too.
- Keep the regex only as a fallback.

**Done when:** a failing pytest inside the build subagent is detected by the loop without parsing prose.

---

## MINOR FIXES

**m1. File naming.** The planner writes `plan.md` (`nodes/planning_subagent.py`); the build prompt, README and `permissions.json` mix `plan.md` and `PLAN.md`. Use `PLAN.md` and `ARCHITECTURE.md` everywhere (code, prompts, permission patterns).

**m2. `make verify` mutates files.** `Makefile` `lint` runs `ruff check . --fix` and `ruff format .` on the whole repo. Add `make check` (`ruff check .`, `ruff format --check .`, `pytest -q`) for the agent to use, and keep `make fix` for humans.

**m3. Verification timeout.** `tools/verification_tools.py:202` uses a fixed 120s subprocess timeout. Make it configurable per command and per tier (default higher for tests and builds).

**m4. Pollution of harness files.** `harness/failures.md` is full of repeated test rows (`rm -rf /`, `git push origin main`) from the test suite. Make tests write to `tmp_path`, clear the file, and add `harness/failures.md`, `harness/progress.md`, `harness/sessions/` to `.gitignore` (keep a template copy).

**m5. Sandbox default.** `agent/../frameworks/deepagents_backend.py` defaults `SANDBOX_MODE` to `local` (host shell), unlike what the README lifecycle describes. Default to `auto` (Docker with fallback) and log which mode is active at startup. Update the README to match reality.

**m6. Unused state fields.** `AgentState` fields `plan_md`, `architecture_md`, `current_stage`, `approved_files`, `generated_files` are never set outside tests. Either wire them into the M3 loop or delete them.

**m7. Whole-agent fallback restarts from scratch.** `DeepAgentsBackend.build_agent` wraps the whole compiled agent with `.with_fallbacks([...])`. If the primary fails mid-run, the fallback agent starts over on the same input. Use model-level retry/fallback (without breaking Deep Agents' `.partition()` requirement noted in the code comment) so a failure retries only the failed model call.

**m8. Prompt-injection filter is too blunt.** `_INJECTION_PATTERNS` in `agent/core.py` uses substring matching, so a request containing `system:` (YAML, logs) or `you are now` is blocked. Match more narrowly (word boundaries, start-of-line, or require surrounding instruction phrasing) and add tests with legitimate code containing those strings.

**m9. Central model config.** `qwen/qwen3.8-27b` and `gpt-oss:120b-cloud` are hard-coded in `core.py`, `agents_md_writer.py`, the planning, design and build files, and `deepagents_backend.py`. Move them into one config module/env file. Add a note in the README that small models are weak at long multi-step tool use, and let the planner and builder use different models.

**m10. `progress.md` is never updated.** It still says "idle / none". Update it from the M3 loop (see M3) and have the build prompt's orientation step read it.

**m11. README accuracy.** After the fixes above, update `README.md`, `CODING_AGENT_IMPLEMENTATION.md` and `TODO_MECHANISM_SPEC.md` so every claim matches what is wired and running (MCP loaded, E2E, Docker default, resume after restart).

**m12. Tests to add.** One test per major fix: permission profiles, `TASK.md` separation, multi-task loop with a blocked task, subagent tool presence, MCP loading with one server down, resume after restart, structured verification result.

---

## Suggested order

1. M1 (permissions) and M2 (AGENTS.md): quick wins, they remove most of the hand-holding.
2. M4 and M5: make the agent have the tools it is supposed to have.
3. M3 and M7: the actual autonomous loop.
4. M6: durability for long runs.
5. Minor fixes m1-m12, then update docs.
