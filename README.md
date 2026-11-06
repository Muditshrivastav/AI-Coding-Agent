# AI Coding Agent

An AI coding agent built as an **execution harness**, not a chat assistant. It plans a request into independently verifiable tasks, executes them one at a time, and continues past blocked tasks. It can expose Git, MCP, and deployment tools; irreversible actions remain permission-gated and are not an automatic completion step.

It is built on **LangGraph + Deep Agents**, loads optional external services through **MCP**, retrieves context from your codebase with **GraphRAG (Neo4j)**, and runs shell commands in the configured sandbox. `SANDBOX_MODE=auto` prefers Docker and falls back to the local workspace shell when Docker is unavailable.

> **Why "harness"?** The model is only one part. The harness is everything around it: instruction files, tools, verification scripts, permission rules, memory and logs. This project is organised around that idea (see [Harness architecture](#harness-architecture)).

---

## Table of contents

1. [Features](#features)
2. [Tech stack and frameworks](#tech-stack-and-frameworks)
3. [How it works](#how-it-works)
4. [Harness architecture](#harness-architecture)
5. [GraphRAG: retrieval before generation](#graphrag-retrieval-before-generation)
6. [Sandbox execution lifecycle](#sandbox-execution-lifecycle)
7. [Project structure](#project-structure)
8. [Getting started](#getting-started)
9. [Usage](#usage)
10. [Configuration](#configuration)
11. [Further documentation](#further-documentation)
12. [Known limitations](#known-limitations)

---

## Features

- **Structured task execution**: a planning agent writes `PLAN.md` and `harness/tasks.json`; the loop verifies each task's `verify_command`, retries/replans failures, and continues after blocked tasks.
- **Checkpointed runs**: a Postgres DSN enables LangGraph's async Postgres checkpointer and restart recovery; without a DSN, checkpoints are in-memory.
- **Optional MCP tools**: GitHub requires a configured token; Chrome DevTools, WebMCP, draw.io, and Render are loaded when available/configured. Failed servers are logged and isolated; `/tools` reports loaded names and status.
- **Command-based verification**: each task's verification command runs independently, and failing output is fed into retries. Browser E2E verification is not currently wired into the task loop.
- **Human-in-the-loop permissions**: `strict` remains the default. The `autonomous` profile allows routine work and still asks before irreversible actions.
- **Optional GraphRAG retrieval** over the repository when Neo4j is configured; RAGAS utilities are not run as part of each task.
- **Sandbox modes**: `auto` (default; Docker with local fallback), `docker` (require Docker), or `local` (host shell).
- **Multiple interfaces**: a VS Code extension, a browser UI, a CLI, and a FastAPI backend.
- **Observability**: LangSmith traces plus `progress.md` and `failures.md` logs.

---

## Tech stack and frameworks

| Layer | Technology | Role |
|---|---|---|
| Agent framework | **Deep Agents** (`deepagents`) | `create_deep_agent`, runnable subagents, filesystem and shell backends |
| Orchestration | **LangGraph**, **LangChain** | State graph, checkpointing, interrupts (HITL), retries |
| Tool protocol | **MCP** (official Python SDK, `mcp`) | `ClientSession`, `stdio_client`, `streamablehttp_client`; tools converted to LangChain `StructuredTool`s |
| Source control | **GitHub MCP** | Loaded when a GitHub token is configured; available operations depend on the remote server |
| Deployment | **Render MCP** | Optional and enabled when configured |
| Browser tooling | **Chrome DevTools MCP**, WebMCP | Optional tools for browser inspection; not used as automated task verification |
| Web search | **Tavily** (`langchain-tavily`, `tavily-python`) | Looks up current docs and APIs |
| Code retrieval | **Neo4j** + `neo4j-graphrag` | Graph of files, functions and classes with vector embeddings |
| Repo ingestion | **LlamaIndex** (`llama-index-readers-github`) | Reads repo files through the GitHub API for ingestion |
| Embeddings and reranking | `sentence-transformers`, `langchain-huggingface` | Embeddings and a cross-encoder reranker |
| Retrieval evaluation | **RAGAS** | Evaluation utilities; not part of per-task verification |
| Observability | **LangSmith** | Optional tracing when configured |
| LLM providers | `langchain-ollama`, `langchain-huggingface` | Ollama chat models and Hugging Face model utilities via LangChain |
| Persistence | **Postgres** (`langgraph-checkpoint-postgres`) | Durable LangGraph checkpoints when a Supabase/Postgres DSN is configured |
| Cross-session memory | **Supabase** (Postgres via `asyncpg`) | Optional session memory and metadata |
| Sandbox | **Docker** (`docker` SDK) | Isolated command execution when available; `auto` falls back to the local shell |
| API layer | **FastAPI** | HTTP and SSE surface over the harness |
| Editor integration | **VS Code extension** + **ACP** (`agent-client-protocol`) | Runs the agent inside VS Code through the extension, which talks to the ACP server |
| Browser UI | Web UI (`ui/`) | Use the agent from the browser, on top of the FastAPI backend |
| Testing and tooling | `pytest`, `ruff`, `uv` | Verification and dependency management |

Requires **Python 3.12+**.

Small or latency-optimized models may struggle with long, multi-step tool use. The planner and builder have separate model settings so you can choose a stronger model for task execution.

---

## How it works

```
User request
     |
     v
+---------------------------------------------------------------+
|  CodingAgentHarness  (run / resume, thread-scoped)            |
|                                                               |
|  1. Plan request       ->  PLAN.md + harness/tasks.json        |
|  2. Execute one task   ->  verify, retry/replan or block       |
|  3. Continue tasks     ->  final status + progress.md          |
+---------------------------------------------------------------+
```

### The task loop

Each task gets a fresh agent invocation and an independent verification command:

```
plan -> pick task -> implement -> verify_command
                             | pass       | fail
                             v            v
                         mark done    retry -> replan -> block
                             |                         |
                             +---- next task <---------+
                                       |
                                       v
                               final verification
```

The default is four attempts per task. Run-level iteration, wall-clock, and token limits are configurable with `AGENT_MAX_TASK_ATTEMPTS`, `AGENT_MAX_ITERATIONS`, `AGENT_MAX_WALL_CLOCK_MINUTES`, and `AGENT_MAX_TOKENS`. The final status is `complete`, `partial`, or `budget_exceeded`; blocked tasks are listed in the result. Browser E2E is not currently part of this loop.

The outer LangGraph state and task list are checkpointed. Task attempts use separate thread IDs, and `harness/progress.md` plus `harness/tasks.json` are updated as the loop advances.

### Entry points

Initialize the harness asynchronously so it can configure the checkpointer and optional MCP clients:

```python
harness = await CodingAgentHarness.create(root_dir=".")
```

- `run(user_request, thread_id)` starts a task run.
- `resume(thread_id, approved=None)` continues a checkpointed run; pass an approval decision when the checkpoint is waiting for one.
- `POST /runs` returns `{thread_id, status: "running"}` immediately. `GET /runs/{thread_id}` returns status and progress.
- `GET /tools` and `/health` expose MCP status and the tool names that actually loaded.

When a Postgres DSN is configured (`supabase_uri`, `SUPABASE_URI`, `DATABASE_URL`, or `SUPABASE_DB_URL`), runs use `AsyncPostgresSaver`; without one, graph checkpoints are in-memory and do not survive restart. Session metadata is stored under `harness/sessions/`.

The planner, designer, and builder are registered as compiled runnable subagents with their required tools. See `TODO_MECHANISM_SPEC.md` for the actual task schema and transitions.

---

## Harness architecture

The harness follows a five-component model. Each component maps to concrete files in this repo.

| # | Component | What it looks like here |
|---|---|---|
| 1 | **System of record** | Permanent `harness/AGENTS.md`, per-request `harness/TASK.md`, `PLAN.md`, `ARCHITECTURE.md`, and `harness/tasks.json`. |
| 2 | **Tools** | Deep Agents filesystem and sandbox shell, plus configured MCP, API, deployment, and repository retrieval tools. |
| 3 | **Feedback and verification** | `make check` (lint, format check, tests) and each task's `verify_command`; no automated browser E2E tier is wired. |
| 4 | **Guardrails and permissions** | `harness/permissions.json` with `strict` and `autonomous` profiles, guarded interrupts, and configurable shell sandbox mode. |
| 5 | **Observability and memory** | LangSmith traces, `harness/progress.md`, `harness/failures.md`, sessions, and optional Postgres memory/checkpoints. |

### Layered view

```
        Interfaces:  VS Code extension (ACP)  |  Browser UI  |  CLI  |  FastAPI
                                    |
                                    v
        +----------------------------------------------------------+
        |                 CodingAgentHarness (agent/)              |
        |      LangGraph runtime: state, checkpoints, interrupts   |
        +----------------------------------------------------------+
              |                       |                        |
              v                       v                        v
      Subagents (nodes/)      Frameworks (frameworks/)     Tools (tools/)
      planning / design /     one wrapper per external     API toolset,
      build                   framework or service         deploy, retrievers
              |                       |
              v                       v
      Guardrails                 External services
      permissions.json           GitHub, Render, Chrome DevTools,
      HITL approvals             Tavily, Neo4j, Supabase, LangSmith
      configured command sandbox (Docker/local)
```

### Guardrails in detail

`permissions.json` defines `strict` and `autonomous` profiles, each with `allow`, `ask`, and `deny` rules:

- **strict** (default): conservative approvals for shell and write actions.
- **autonomous**: routine shell, write, and verification actions are allowed; irreversible commands and deletes still ask. This profile is accepted only with `SANDBOX_MODE=docker` or `auto`; local mode falls back to strict.
- **deny rules**: never permitted, such as reading `.env` or `secrets/**` and `rm -rf`.

Permanent harness rules require tests to remain intact and prohibit pushing or deploying without explicit approval.

### Verification

`make check` runs read-only lint, format checks, and tests. The task loop runs each task's configured command and checks its exit status independently. Chrome DevTools MCP is available for browser inspection when it loads, but automated E2E verification is not implemented.

On failure, the real stderr or traceback goes into the next generation call. There are no blind retries.

Framework integrations generally live under `frameworks/`; existing direct integrations elsewhere are documented and tested at their call sites.

---

## GraphRAG: retrieval before generation

Before the build subagent writes code, it retrieves relevant code from the target repository. Two independent pipelines share one Neo4j store.

**Ingestion**

1. Acquire the repo (`git clone` for a working checkout, or the LlamaIndex GitHub reader for API-only ingestion).
2. Parse each file with `ast` (Python) or `tree-sitter` (other languages). A chunk is always a whole function, class or module block, never an arbitrary slice of text.
3. Store each unit as a graph node with three layers: structure (`CONTAINS`, `CALLS`, `IMPORTS`, `INHERITS` edges), a vector embedding of its source, and metadata (project, path, language, commit SHA, line range).
4. Re-ingest incrementally on push.

**Retrieval and generation**

1. Metadata-filtered vector search (project, language, path scope), top ~20 candidates.
2. Cross-encoder reranking down to the top ~5.
3. Context assembly with required `[path:line]` citations.
4. Generation under a "cite it or say there is insufficient context" rule.
5. RAGAS faithfulness check. Below the threshold, retry with a narrowed query. Passing output goes on to verification.

Retrieval quality lowers how often bad code is generated, while verification catches it when it happens anyway. Neither replaces the other. Full design in `GRAPHRAG_DESIGN.md`.

---

## Sandbox execution lifecycle

The current Docker backend runs commands in a container with the workspace mounted, so file changes persist in that workspace. It does not create an ephemeral clone or implement the network/resource restrictions described in the original sandbox proposal. `SANDBOX_MODE=auto` falls back to a host-local shell if Docker is unavailable; set `SANDBOX_MODE=docker` to require Docker. Do not treat the sandbox as a security boundary for untrusted workloads.

---

## Project structure

```
agent/          Orchestrator: CodingAgentHarness, shared state
frameworks/     One wrapper class per external framework (LangGraph, Deep Agents,
                MCP client, GitHub/Render/Chrome DevTools MCP, Tavily,
                LangSmith, RAGAS, Supabase, Docker sandbox, VS Code workspace)
nodes/          Planning, design and build subagents
tools/          API toolset, deploy toolset, codebase and docs retrievers
graphrag/       Neo4j GraphRAG ingestion and retrieval
harness/        AGENTS.md, TASK.md, permissions.json, task/progress/failure state
interfaces/     api_server.py (FastAPI), acp_server.py (ACP, used by the VS Code extension)
skills/         Reusable, project-specific procedures the agent can reference
ui/             Browser UI
vscode-extension/   VS Code extension (ACP client)
tests/          pytest suite
cli.py          Command-line entry point
Makefile        verify, lint, test, run-api
Dockerfile.sandbox  Optional sandbox image
pyproject.toml, uv.lock   Dependencies
```

---

## Getting started

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), and a running Neo4j instance if you use GraphRAG. Docker is optional in `auto` mode.

```bash
git clone https://github.com/Muditshrivastav/AI-Coding-Agent.git
cd AI-Coding-Agent

# Install dependencies
uv sync

# Configure environment
cp .env.example .env     # then fill in your keys

# Optional: build the sandbox image
docker build -f Dockerfile.sandbox -t ai-coding-agent-sandbox:latest .
```

---

## Usage

**CLI**

```bash
# Start a task
python cli.py "Build a REST API for a todo app"

# Choose the workspace root and thread explicitly
python cli.py "Add login endpoint" --root-dir ./my-project --thread-id my-thread

# Resume a run that paused for approval
python cli.py "" --thread-id my-thread --resume --approve
```

The workspace root resolves in this order: `--root-dir`, then the VS Code workspace, then the git root, then the current directory.

**VS Code**

The agent is already integrated into VS Code through the extension in `vscode-extension/`. The extension connects to the agent over ACP (`interfaces/acp_server.py`), so you can use it from the editor without leaving your workspace.

**Browser UI**

The agent also has a browser-based UI (in `ui/`), which works against the FastAPI backend below.

**FastAPI backend**

```bash
make run-api        # http://localhost:8000
```

**Verification**

```bash
make check          # ruff check, ruff format --check, pytest (read-only)
make fix            # auto-fix lint/format issues
```

---

## Configuration

Copy `.env.example` to `.env`. It covers:

| Group | Variables |
|---|---|
| Search and deploy | `tavily_api_key`, `render_api_key` |
| GitHub OAuth app | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GITHUB_REDIRECT_URI` |
| GitHub MCP and ingestion | `GITHUB_PERSONAL_ACCESS_TOKEN`, `GITHUB_PAT`, or `GITHUB_TOKEN` |
| Postgres checkpoints and memory | `supabase_uri`, `SUPABASE_URI`, `DATABASE_URL`, or `SUPABASE_DB_URL` |
| LangSmith tracing | `LANGSMITH_*` and `LANGCHAIN_*` variables |
| Neo4j GraphRAG | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |
| Permissions and sandbox | `PERMISSION_PROFILE`, `SANDBOX_MODE` |
| Task-loop budgets | `AGENT_MAX_TASK_ATTEMPTS`, `AGENT_MAX_ITERATIONS`, `AGENT_MAX_WALL_CLOCK_MINUTES`, `AGENT_MAX_TOKENS` |

Never commit `.env`. `strict` is the default permission profile; configure profiles in `harness/permissions.json`.

---

## Further documentation

| File | Contents |
|---|---|
| `CODING_AGENT_IMPLEMENTATION.md` | Original architecture proposal; some parts are not wired and it is not the runtime source of truth |
| `GRAPHRAG_DESIGN.md` | GraphRAG ingestion, retrieval, reranking and anti-hallucination design |
| `SANDBOX_EXECUTION_LIFECYCLE.md` | Sandbox implementation limits and proposed isolation lifecycle |
| `TODO_MECHANISM_SPEC.md` | Structured task model, verification, retries, and statuses |

---

## Known limitations

- Automated browser E2E verification is not implemented.
- Durable graph checkpointing requires a configured Postgres DSN; otherwise checkpoints are in-memory.
- The FastAPI service does not currently enforce API authentication.
- `auto` sandbox mode may fall back to the host shell if Docker is unavailable.
