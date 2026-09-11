# Coding Agent — Full Implementation Spec

Goal: an autonomous coding agent, built as an execution **harness** (not a chat assistant), able to plan, design, and build real software projects end to end — commit, push, and deploy included. Targets parity with Cursor/Claude Code/Codex-style agents. This is the first agent build in the broader project roadmap.

---

## 1. Core stack

| Layer | Choice |
|---|---|
| Agent framework | Deep Agents (`create_deep_agent`, subagents, filesystem middleware, todo middleware) |
| Orchestration | LangChain + LangGraph (state graph, checkpointing, interrupts) |
| Observability | LangSmith (tracing every tool call and subagent delegation) |
| Retrieval evaluation | RAGAS (scores the dev subagent's RAG retrieval before code generation) |
| Web search | Tavily (`langchain_tavily.TavilySearch`) |
| Browser control / E2E testing | Chrome DevTools MCP |
| Source control | GitHub MCP (official remote server, write-capable) |
| Deployment | Vercel MCP / Render MCP (official servers — **read-only** for deploy actions; actual triggers go through their CLIs as shell tools) |
| Sandbox / execution | Deep Agents `LocalShellBackend`, scoped to `root_dir` |
| MCP integration | Official MCP Python SDK (`ClientSession`, `stdio_client`, `streamablehttp_client`) — no third-party adapter layer |
| Editor integration | `deepagents-acp` (ACP server) + a VS Code ACP client extension |
| Serialization | TOON (Token Object-Oriented Notation) — used to minimize token consumption in state/tool payloads |
| Cross-session memory | Supabase (Postgres, via `asyncpg`) — persistent memory beyond a single LangGraph thread |
| Approval UI | Streamlit app with HITL approval buttons (primary), ACP permission requests (editor-side) |
| API layer | FastAPI — HTTP/SSE surface over the harness, used by the Streamlit app, external clients, and webhook triggers |

---

## 2. The harness, not just an agent

Per the Harness Engineering model (Srinivasan), a harness is the full system of artifacts around the model — not a wrapper class. Five components, each mapped to a concrete piece of this build:

| # | Component | What it looks like here |
|---|---|---|
| 1 | System of record | `AGENTS.md` (root), `PLAN.md`, `ARCHITECTURE.md`, a `skills/` section, an `AGENT.md` memory section |
| 2 | Tools | LocalShellBackend, GitHub MCP, Vercel/Render MCP, Tavily, Chrome DevTools MCP, custom API toolset |
| 3 | Feedback & verification | `make verify` (lint → format → test), Chrome DevTools MCP for E2E, RAGAS for retrieval quality |
| 4 | Guardrails & permissions | `permissions.json` (allow/ask/deny), HITL approval gate, root_dir sandboxing |
| 5 | Observability & memory | LangSmith traces, Deep Agents `write_todos`, `progress.md`, `failures.md`, Supabase (Postgres) |

The rest of this document is organized around these five components, plus the orchestration layer that ties them together.

---

## 3. Directory structure (OOP, one framework per file)

```
agent/
  core.py                      # CodingAgentHarness — the only entry point (run/resume)
  state.py                     # AgentState (TypedDict)
  harness.py                   # ties frameworks + subagents together

frameworks/
  langgraph_runtime.py         # LangGraphRuntime — StateGraph, checkpointing
  deepagents_backend.py        # DeepAgentsBackend — create_deep_agent + LocalShellBackend
  mcp_client.py                # MCPClientWrapper — official MCP SDK, stdio + HTTP transports
  tavily_search.py             # TavilySearchTool
  chrome_devtools_mcp.py       # ChromeDevToolsMCP (MCPClientWrapper, stdio)
  github_mcp.py                # GitHubMCP (MCPClientWrapper, HTTP)
  vercel_mcp.py                # VercelMCP (MCPClientWrapper, HTTP, read-only)
  render_mcp.py                # RenderMCP (MCPClientWrapper, HTTP, read-only)
  langsmith_tracer.py          # LangSmithTracer
  ragas_evaluator.py           # RagasEvaluator
  supabase_memory.py           # SupabaseMemory — cross-session persistent memory (Postgres, asyncpg)

nodes/
  planning_subagent.py         # writes PLAN.md
  design_subagent.py           # writes ARCHITECTURE.md
  build_subagent.py            # implements, verifies, commits, deploys

tools/
  api_tools.py                 # APIToolset — custom REST APIs as StructuredTools
  deploy_tools.py              # DeployToolset — vercel_deploy via CLI/shell
  codebase_retriever.py        # CodebaseRetriever — embeddings over target repo
  docs_retriever.py            # DocsRetriever — embeddings over external docs

interfaces/
  streamlit_app.py             # HITL approval UI
  acp_server.py                # exposes the harness over ACP for VS Code
  api_server.py                # FastAPI — HTTP/SSE surface over the harness

harness/
  AGENTS.md                    # system of record, root instruction file
  permissions.json             # guardrails: allow / ask / deny
  progress.md                  # observability: current stage, done/pending
  failures.md                  # observability: triaged failure log

cli.py                          # command-line entry point
Makefile                        # `make verify` — single verification command
```

**Build order** (locked): `agent/core.py` → `interfaces/streamlit_app.py` → `interfaces/acp_server.py` → `cli.py` → subagents.

**Rule enforced throughout**: each file under `frameworks/` wraps exactly one external framework/API. Nothing outside `frameworks/` imports a raw framework directly — only the class interface.

---

## 4. Shared state

```python
# agent/state.py
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_request: str
    plan_md: str
    architecture_md: str
    retrieved_context: list[dict]      # tagged {"source": "codebase"|"external_docs", ...}
    generated_files: list[dict]        # staged {path, content}, pending approval
    approved_files: list[str]
    todos: list[dict]                  # Deep Agents' write_todos state
    verify_attempts: int
    verify_passed: bool
    messages: Annotated[list, add_messages]
    current_stage: Literal["planning", "design", "build", "verify", "done"]
```

A single shared `State` schema — all subagents operate as delegated tasks against this one state, not separate graphs.

---

## 5. Component 1 — System of record

**`harness/AGENTS.md`** (root, kept under a page, read before any subagent plans or builds):

```markdown
# Project: <name>
<one sentence on what this coding agent builds and for whom>

## Structure
- `agent/` orchestrator + harness code
- `nodes/` planning, design, build subagent prompts
- `frameworks/` one class per external framework
- `tools/` custom API-backed tools

## Commands
- Verify (all checks): `make verify`
- Lint: `ruff check . --fix`
- Format: `ruff format .`
- Test: `pytest -q`

## Conventions
1. Each file wraps exactly one framework — never mix LangGraph and Deep Agents calls in one file.
2. Subagents read PLAN.md / ARCHITECTURE.md via read_file — never re-derive scope themselves.
3. All shell/file-write actions go through LocalShellBackend — nothing executes outside root_dir.

## Do not
- Edit or delete a test to make a run pass.
- Push to GitHub or trigger a deploy without an approval checkpoint.
- Add a new dependency without flagging it in PLAN.md first.
```

Per-run artifacts (not standing instructions, written fresh each build):
- **`PLAN.md`** — written by planning-subagent: scope, constraints, acceptance criteria, file/module list.
- **`ARCHITECTURE.md`** — written by design-subagent: components, data flow, module boundaries, tech choices.

A `skills/` directory and an `AGENT.md` memory section extend this further with reusable, project-specific procedures the agent can reference — same "read before you act" principle as `AGENTS.md`, scoped to narrower recurring tasks.

---

## 6. Component 2 — Tools

### 6.1 Deep Agents backend (filesystem + shell)

```python
# frameworks/deepagents_backend.py
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

class DeepAgentsBackend:
    def __init__(self, root_dir: str = "."):
        self._backend = LocalShellBackend(root_dir=root_dir, env={"PATH": "/usr/bin:/bin"})

    def build_agent(self, model: str, subagents: list, system_prompt: str, checkpointer=None):
        return create_deep_agent(
            model=model,
            backend=self._backend,
            subagents=subagents,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
```

File writes are **staged, not committed** — the build subagent calls `propose_write`-equivalent behavior (write_file is interrupted before execution per `permissions.json`), and only executes after HITL approval.

### 6.2 MCP client wrapper (official SDK, one class, both transports)

```python
# frameworks/mcp_client.py
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.tools import StructuredTool
from pydantic import create_model

class MCPClientWrapper:
    """Wraps one MCP server (stdio or HTTP) via the official SDK and converts
    its tools into LangChain StructuredTools."""

    def __init__(self, transport: str, **kwargs):
        self._transport = transport  # "stdio" | "http"
        self._kwargs = kwargs
        self._session: ClientSession | None = None
        self._stack = AsyncExitStack()

    async def connect(self):
        if self._transport == "stdio":
            params = StdioServerParameters(
                command=self._kwargs["command"], args=self._kwargs.get("args", []),
                env=self._kwargs.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(url=self._kwargs["url"], headers=self._kwargs.get("headers"))
            )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def close(self):
        await self._stack.aclose()

    def _make_tool(self, mcp_tool) -> StructuredTool:
        schema = mcp_tool.inputSchema or {"type": "object", "properties": {}}
        fields = {name: (str, ...) for name in schema.get("properties", {})}
        ArgsModel = create_model(f"{mcp_tool.name}_Args", **fields) if fields else None

        async def _call(**kwargs):
            result = await self._session.call_tool(mcp_tool.name, arguments=kwargs)
            return "\n".join(b.text for b in result.content if hasattr(b, "text"))

        return StructuredTool.from_function(
            coroutine=_call, name=mcp_tool.name,
            description=mcp_tool.description or "", args_schema=ArgsModel,
        )

    async def get_tools(self) -> list[StructuredTool]:
        result = await self._session.list_tools()
        return [self._make_tool(t) for t in result.tools]
```

`CodingAgentHarness` owns one `MCPClientWrapper` instance per server and keeps it open for the process lifetime — the session isn't a fire-and-forget connection, it needs to stay alive across tool calls, and gets `.close()`d on shutdown.

### 6.3 GitHub MCP (write-capable)

```python
# frameworks/github_mcp.py
class GitHubMCP:
    def __init__(self, pat: str):
        self._client = MCPClientWrapper(
            transport="http",
            url="https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {pat}"},
        )
    async def get_tools(self) -> list:
        await self._client.connect()
        return await self._client.get_tools()  # commit, push, branch, PR
```

### 6.4 Vercel / Render MCP (read-only status + logs)

```python
# frameworks/vercel_mcp.py
class VercelMCP:
    def __init__(self, oauth_token: str):
        self._client = MCPClientWrapper(
            transport="http", url="https://mcp.vercel.com",
            headers={"Authorization": f"Bearer {oauth_token}"},
        )
    async def get_tools(self) -> list:
        await self._client.connect()
        return await self._client.get_tools()  # deployment status, logs, project info
```
Render MCP (`mcp.render.com`) follows the same shape. **Neither triggers a deploy** — actual deploy triggering is a shell tool (see 6.6), or relies on git-push auto-deploy once GitHub MCP pushes.

### 6.5 Tavily + Chrome DevTools MCP

```python
# frameworks/tavily_search.py
from langchain_tavily import TavilySearch
class TavilySearchTool:
    def __init__(self, max_results=5, topic="general"):
        self._tool = TavilySearch(max_results=max_results, topic=topic)
    def as_langchain_tool(self): return self._tool

# frameworks/chrome_devtools_mcp.py
class ChromeDevToolsMCP:
    def __init__(self):
        self._client = MCPClientWrapper(transport="stdio", command="npx", args=["chrome-devtools-mcp@latest"])
    async def get_tools(self) -> list:
        await self._client.connect()
        return await self._client.get_tools()
```

### 6.6 Custom API toolset + deploy toolset

```python
# tools/api_tools.py — StructuredTool.from_function bound to instance methods (state: tokens, base URLs)
class APIToolset:
    def get_tools(self) -> list[StructuredTool]: ...  # e.g. create_github_issue, check_lint_status

# tools/deploy_tools.py — actual deploy trigger, via shell (CLI), not MCP
class DeployToolset:
    def _vercel_deploy(self, project_dir: str, prod: bool = False) -> dict:
        return self._shell.execute(f"cd {project_dir} && vercel deploy {'--prod' if prod else ''} --yes")
    def get_tools(self) -> list[StructuredTool]: ...
```

### 6.7 Retrievers (codebase + external docs)

Two separate vector stores, tagged by `source` so RAGAS can score them independently:
- **Codebase retriever** — built fresh per session/target repo (walk → chunk by function/class → embed).
- **Docs retriever** — persistent, refreshed only on dependency version bumps.

---

## 7. Component 3 — Feedback and verification

Three tiers, one entry point:

```makefile
# Makefile
verify:
	ruff check . --fix
	ruff format .
	pytest -q
```

- **Tier 1 (static)** — lint/format, runs after every file edit.
- **Tier 2 (unit/integration)** — `pytest`, runs before every commit.
- **Tier 3 (end-to-end)** — Chrome DevTools MCP drives the actual built app; required before a feature is declared done, not optional.

**RAGAS** scores the retrieval step specifically (codebase-context relevance/faithfulness vs. external-docs relevance) — a separate signal from LangSmith's trajectory tracing.

**Non-negotiable rule** (in `AGENTS.md`, enforced, not just requested): the agent may never edit or delete a test to make a run pass.

**Retry mechanics**: on a `make verify` failure, the actual stderr/traceback is fed back into the next generation call — never a blind retry. Capped at 2–3 attempts per todo item, then escalated to HITL.

---

## 8. Component 4 — Guardrails and permissions

```json
// harness/permissions.json
{
  "allow": ["Bash(pytest*)", "Bash(ruff*)", "Read(src/**)", "Edit(src/**)"],
  "ask": ["Bash(git push*)", "vercel_deploy", "mcp__github__*"],
  "deny": ["Read(.env)", "Read(secrets/**)", "Bash(rm -rf*)"]
}
```

- `LocalShellBackend` is scoped to `root_dir` — the sandbox boundary. Deny rules are a filter on top, not the sandbox itself.
- `ask` entries populate `interrupt_before` on the corresponding Deep Agents tools — this is the HITL gate. Approvals surface via the Streamlit app (primary) and via ACP permission requests when connected through VS Code.
- Escalation ladder for repeated mistakes: 1st time → fix in session. 2nd time → line in `AGENTS.md`. 3rd time → convert to a `deny` rule or a `make verify` check.

---

## 9. Component 5 — Observability and memory

- **LangSmith** — traces every tool call, subagent delegation, and retrieval call as separate spans.
- **`write_todos`** (Deep Agents native) — the build subagent's task queue; also the harness's live progress record via `AgentState`.
- **`progress.md`** — human-readable derived view of todo state, rewritten after each item completes; read at the start of every resumed run (orientation sequence).
- **`failures.md`** — one entry per failed run, classified against the triage table (system of record / tools / verification / feedback / guardrails / memory).
- **Supabase (Postgres)** — persistent memory that survives beyond a single LangGraph thread/checkpointer, keyed by project, for anything that should carry across separate build sessions rather than just across interrupts within one. Accessed via `asyncpg` to stay consistent with the harness's fully async design; LangGraph's `AsyncPostgresSaver` can optionally share the same database for thread checkpointing.

```python
# frameworks/supabase_memory.py
import asyncpg, json

class SupabaseMemory:
    def __init__(self, dsn: str):
        self._dsn = dsn  # Supabase connection string (session pooler recommended)
        self._pool: asyncpg.Pool | None = None

    async def init(self):
        self._pool = await asyncpg.create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    project_id TEXT, key TEXT, value JSONB,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (project_id, key)
                )""")

    async def save(self, project_id: str, key: str, value: dict):
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO agent_memory (project_id, key, value) VALUES ($1, $2, $3)
                   ON CONFLICT (project_id, key) DO UPDATE SET value = $3, updated_at = now()""",
                project_id, key, json.dumps(value),
            )

    async def load(self, project_id: str, key: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM agent_memory WHERE project_id = $1 AND key = $2", project_id, key
            )
            return json.loads(row["value"]) if row else None
```

**Note on Supabase's free tier**: projects pause after 7 days of inactivity and need a manual unpause from the dashboard — this can strand an agent that returns to a project after a break. Either use a paid tier (always-on) or add a lightweight scheduled ping to keep the project active if staying on the free tier during early development.
- **TOON serialization** — used across state and tool payloads to minimize token consumption, in line with the project's stated goal of keeping this a low-token-footprint agent.

---

## 10. Orchestration — subagents and the build loop

### 10.1 Subagent definitions

```python
planning_subagent = {
    "name": "planning-agent",
    "description": "Use FIRST for any new build request. Writes PLAN.md.",
    "system_prompt": "Write a clear implementation plan as PLAN.md. No code, no architecture.",
}

design_subagent = {
    "name": "design-agent",
    "description": "Use AFTER planning-agent. Reads PLAN.md, writes ARCHITECTURE.md.",
    "system_prompt": "Read PLAN.md. Write ARCHITECTURE.md: components, data flow, file structure.",
}

def build_dev_subagent(
    all_tools: list | None = None,
    external_tools: ExternalToolsManager | None = None,
    mcp_clients: list[MCPClientWrapper] | None = None,
) -> dict:
    return {
        "name": "build-agent",
        "description": "Use AFTER design-agent. Implements, verifies, commits, deploys.",
        "system_prompt": BUILD_SYSTEM_PROMPT,
        "tools": combined_tools,
        "factory": make_build_subagent,
        "external_tools": external_tools,
        "mcp_clients": mcp_clients or [],
    }
```

### 10.2 The build loop (per todo item)

```
Orientation (read AGENTS.md, PLAN.md, ARCHITECTURE.md, progress.md, git status)
        ↓
Decompose into todos (write_todos, derived from ARCHITECTURE.md)
        ↓
   ┌──→ Retrieve + generate (codebase RAG + docs RAG → write_file, staged)
   │        ↓
   │    Verify (make verify)
   │        ↓ pass                    ↓ fail
   │    Commit + update todo      (loop back with error text)
   │        ↓
   └── more todos? ── yes ──┘
        ↓ no
   End-to-end check (Chrome DevTools MCP)
        ↓
   Feature complete
```

### 10.3 Orchestrator entry points

```python
class CodingAgentHarness:
    def __init__(self, root_dir: str, model: str, tools: list, tracer):
        self._backend = LocalShellBackend(root_dir=root_dir)
        self._checkpointer = MemorySaver()
        self._agent = create_deep_agent(
            model=model, backend=self._backend,
            subagents=[planning_subagent, design_subagent, build_dev_subagent(tools)],
            checkpointer=self._checkpointer,
        )

    async def run(self, user_request: str, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._agent.ainvoke({"messages": [{"role": "user", "content": user_request}]}, config=config)
        if result.get("__interrupt__"):
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}
        return {"status": "complete", "result": result}

    async def resume(self, thread_id: str, approved: bool):
        config = {"configurable": {"thread_id": thread_id}}
        return await self._agent.ainvoke(Command(resume={"approved": approved}), config=config)
```

`run()` and `resume()` are the only two entry points — both thread-scoped, so an interrupted or crashed build resumes from exactly where it stopped instead of restarting planning→design→build.

---

## 11. Editor integration — VS Code via ACP

```python
# interfaces/acp_server.py
from acp import run_agent
async def main():
    agent = harness.build()  # the same CodingAgentHarness agent
    await run_agent(agent, name="my-coding-harness")
```

```json
// VS Code settings — custom ACP agent entry
{
  "acp.agents": {
    "my-coding-harness": { "command": "python", "args": ["interfaces/acp_server.py"], "cwd": "${workspaceFolder}" }
  }
}
```

Open decision: whether HITL interrupts surface through ACP's own permission-request mechanism (approvals as VS Code prompts) or continue to resolve through the Streamlit app as the single approval channel, with ACP only reflecting status.

---

## 12. API layer — FastAPI backend

A single FastAPI app in front of `CodingAgentHarness` — this is what the Streamlit app, any external client, and GitHub webhook triggers all talk to, instead of each interface embedding its own copy of the harness.

```python
# interfaces/api_server.py
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.core import CodingAgentHarness

app = FastAPI(title="Coding Agent Harness API")
harness: CodingAgentHarness = None  # set in startup, one instance shared across requests

class RunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None

class ResumeRequest(BaseModel):
    approved: bool

@app.on_event("startup")
async def startup():
    global harness
    harness = CodingAgentHarness(root_dir=".", model="anthropic:claude-sonnet-4-6", tools=[], tracer=None)

@app.post("/runs")
async def start_run(req: RunRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    result = await harness.run(req.user_request, thread_id=thread_id)
    return {"thread_id": thread_id, **result}

@app.get("/runs/{thread_id}")
async def get_run_status(thread_id: str):
    # reads progress.md / todo state for this thread, not a fresh invoke
    state = await harness.get_state(thread_id)
    if state is None:
        raise HTTPException(404, "unknown thread_id")
    return state

@app.post("/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest):
    result = await harness.resume(thread_id, approved=req.approved)
    return {"thread_id": thread_id, **result}

@app.get("/runs/{thread_id}/stream")
async def stream_run(thread_id: str):
    async def event_source():
        async for update in harness.stream(thread_id):   # yields todo/progress updates
            yield f"data: {update.model_dump_json()}\n\n"
    return StreamingResponse(event_source(), media_type="text/event-stream")

@app.post("/webhooks/github")
async def github_webhook(payload: dict):
    # e.g. trigger a re-verify or notify on CI status change for a thread's PR
    ...
```

**How this fits the rest of the stack:**

- **Streamlit app** stops holding its own harness instance and instead calls `POST /runs`, polls or subscribes to `/runs/{thread_id}/stream`, and calls `/runs/{thread_id}/resume` for HITL approvals — this is what makes the Streamlit UI a thin client rather than a second copy of the orchestration logic.
- **`/runs/{thread_id}/stream`** is the natural place to surface the `write_todos` progress from Component 5 — each todo status change becomes an SSE event, so any client (Streamlit, a future web UI, the VS Code ACP panel) gets live progress without polling `progress.md` on disk.
- **The ACP server (§11) can stay a separate process** — it doesn't need to go through FastAPI, since ACP already has its own transport. But if you'd rather have one process own the harness instance and have both ACP and FastAPI talk to it, `acp_server.py` can call the same `CodingAgentHarness` object FastAPI's `startup` hook creates, rather than each constructing its own.
- **`run()` stays thread_id-scoped** exactly as designed in §10.3 — FastAPI just adds the HTTP surface, it doesn't change the underlying resume/interrupt semantics.

One thing worth deciding: does the API need auth (API keys, since this can trigger real GitHub pushes and deploys), or is it assumed to run behind a private network/VPN for now — this determines whether an auth middleware belongs in this file or is out of scope for v1.

---

## 13. Open items / not yet finalized

- Multi-project concurrency: single build at a time (v1) vs. multiple parallel threads.
- ACP permission surfacing (see §11).
- Exact `deepagents` `tool_configs`/`interrupt_before` parameter names — confirm against the installed package version before wiring `permissions.json` into it.
- Full `skills/` directory contents and `AGENT.md` memory section format — structure named, content not yet drafted.
