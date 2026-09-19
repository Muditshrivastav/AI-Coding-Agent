# Todo Mechanism — Implementation Spec

**Purpose**: Gives a long-running coding agent a persistent, self-maintained task list so it decomposes multi-step work explicitly instead of attempting everything in one generation, and so its "current focus" survives across many tool calls without drifting out of context (the recitation pattern — restating the objective keeps it in recent context instead of buried behind many tool calls).

This mechanism replaces a static, file-derived task list (e.g. iterating `ARCHITECTURE.md`'s module list directly) with a dynamic, agent-maintained one that can grow or shrink mid-build as work is discovered.

---

## 1. The tool

One tool. Full-list replacement on every call — **not** an append/diff operation.

```python
def write_todos(todos: list[dict]) -> str:
    """
    Overwrites the agent's ENTIRE todo list with the list provided.
    Call this to create the initial breakdown of a task, and again
    every time a todo's status changes.

    Each todo item is a dict:
        {
            "content": str,   # description of the task
            "status": str,    # "pending" | "in_progress" | "completed"
        }

    Returns a confirmation string.
    """
```

**Critical implementation detail**: this replaces the whole list every time it's called — the model must pass the complete, updated list including unchanged items, not a delta. This keeps the tool implementation simple (no diffing logic) and forces the model to actively re-state its full plan on every update, which is itself part of what keeps the objective "recited" in context rather than allowed to decay.

---

## 2. State

Add one field to the agent's persisted state (works with any LangGraph-style `TypedDict` state, or equivalent):

```python
class AgentState(TypedDict):
    # ...existing fields...
    todos: list[dict]   # [{"content": ..., "status": ...}, ...]
```

Because this field lives in the same state object that gets checkpointed, it survives an interrupt/resume cycle automatically — no separate persistence mechanism is needed for that.

---

## 3. Middleware — auto-inject current list into every model call

This is the part that makes the mechanism more than "just a tool" — a wrapping layer that runs before every model turn, so the model doesn't need to call a "read my todos" tool; its current list is re-injected into context automatically.

```python
class TodoListMiddleware:
    def before_model_call(self, state: dict, messages: list) -> list:
        todos = state.get("todos")
        if todos:
            rendered = "\n".join(
                f"- [{'x' if t['status'] == 'completed' else ' '}] {t['content']}"
                + (" (in progress)" if t["status"] == "in_progress" else "")
                for t in todos
            )
            messages.append({
                "role": "system",
                "content": f"Current todo list:\n{rendered}",
            })
        return messages
```

This injected block is the actual mechanism behind "recitation" — the current objective and progress stay in recent context on every single turn, rather than being buried under accumulated tool-call history.

---

## 4. Prompting rules

The tool has no value without behavioral constraints on when and how it's used. These mirror the guidance behind Claude Code's `TodoWrite` tool, which this pattern is based on:

1. **Use it only for tasks with 3+ distinct steps, or genuine complexity.** Skip it for single-step or trivial tasks — creating a todo list for "fix this typo" is overhead, not help.
2. **Exactly one todo may be `in_progress` at a time.** Never mark multiple simultaneously — this is what keeps "current focus" singular and unambiguous for both the agent and anything reading its state externally.
3. **Mark a todo `completed` immediately after finishing it — not batched at the end of a session.** Batching defeats the purpose; progress tracking must be live, not retroactive.
4. **The list is not fixed at creation time.** If new sub-steps are discovered mid-work, call `write_todos` again with the expanded list.
5. **Never mark a todo `completed` based on the model's own assessment alone.** Completion must be gated by an external verification result (tests passing, lint passing) where verification is available — not by the model believing it's "probably done."

---

## 5. Integration into a build loop

```
Read the project's plan/architecture context
        ↓
Call write_todos([...])                          ← initial decomposition
        ↓
   ┌──→ Pick the one "in_progress" todo
   │    (or promote the next "pending" → "in_progress" via write_todos)
   │        ↓
   │    Retrieve relevant context → generate code for this todo only
   │        ↓
   │    Run verification (lint / tests / build)
   │        ↓ pass                           ↓ fail
   │    write_todos([...this one             write_todos unchanged;
   │    marked "completed", next             retry generation with
   │    "pending" → "in_progress"])          the verification error text
   │        ↓                                       │
   │    Derive a human-readable progress file  ◄────┘ (on fail, no progress update)
   │    from current todos state
   │        ↓
   └── more "pending" todos remain? ── yes ──┘
        ↓ no
   All todos "completed" → run any final end-to-end check → done
```

**Key properties this preserves:**
- **One unit of work per generation call** — never attempt the whole task in one shot.
- **A working state to fall back to** — each completed todo should correspond to a committed, verified increment, not an in-progress, unverified change.
- **Live observability** — an external system (a UI, a log, a progress file) can read `state["todos"]` at any point mid-run and know exactly what's done, what's next, and what's currently being worked on, without needing to parse tool-call history.

---

## 6. What this mechanism does NOT do

- It does not decide **what** the todos should be — that still comes from whatever plan/architecture context the agent has been given (e.g., a design document, a user's stated requirements). `write_todos` is the tracking and re-statement mechanism, not the planning source.
- It is not a replacement for actual verification — a todo marked `completed` is only as trustworthy as the check that gated it. If no verification step exists for a given todo, treat its "completed" status as provisional.
- It is not meant for micromanaging trivial actions — a single `write_file` call inside completing one todo does not need its own todo entry.
