# AI Coding Agent

An autonomous coding agent built as an **execution harness**, not a chat assistant. Given a software task, it plans the work, designs the architecture, implements it, verifies it with lint/tests/E2E checks, and commits, pushes and deploys, with a human approval gate before anything irreversible.

It is built on **LangGraph + Deep Agents**, talks to external services through **MCP**, retrieves context from your codebase with **GraphRAG (Neo4j)**, and runs shell commands in a **Docker sandbox**.

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
12. [Open items](#open-items)

---

## Features

- **Three-stage multi-agent pipeline**: a planning subagent writes `PLAN.md`, a design subagent writes `ARCHITECTURE.md`, and a build subagent implements, verifies, commits and deploys.
- **Single shared LangGraph state** across all three stages, with checkpointing so an interrupted run resumes where it stopped.
- **MCP tool access** to GitHub, Vercel, Render and Chrome DevTools via the official MCP Python SDK.
- **Automated verification loop**: lint, format, unit tests, then end-to-end checks in a real browser. Failures feed the actual error text into the next attempt.
- **Human-in-the-loop approval** before file writes, `git push` and deploys, configured through `permissions.json`.
- **GraphRAG retrieval** over your repository (AST-based chunking, Neo4j graph and vectors, cross-encoder reranking), with RAGAS faithfulness scoring to catch ungrounded code.
- **Docker sandbox** for shell execution, so commands never touch your real repository directly.
- **Token-saving design**: TOON serialization for state and tool payloads.
- **Multiple interfaces**: a VS Code extension, a browser UI, a CLI, and a FastAPI backend.
- **Observability**: LangSmith traces plus `progress.md` and `failures.md` logs.

---

## Tech stack and frameworks

| Layer | Technology | Role |
|---|---|---|
| Agent framework | **Deep Agents** (`deepagents`) | `create_deep_agent`, subagents, filesystem and shell backends, todo middleware |
| Orchestration | **LangGraph**, **LangChain** | State graph, checkpointing, interrupts (HITL), retries |
| Tool protocol | **MCP** (official Python SDK, `mcp`) | `ClientSession`, `stdio_client`, `streamablehttp_client`; tools converted to LangChain `StructuredTool`s |
| Source control | **GitHub MCP** | Branch, commit, push, pull request operations |
| Deployment | **Vercel MCP**, **Render MCP** | Read-only status and logs. Deploy triggers go through the CLI as shell tools |
| Browser / E2E | **Chrome DevTools MCP** | Runs and tests the built frontend in a real browser |
| Web search | **Tavily** (`langchain-tavily`, `tavily-python`) | Looks up current docs and APIs |
| Code retrieval | **Neo4j** + `neo4j-graphrag` | Graph of files, functions and classes with vector embeddings |
| Repo ingestion | **LlamaIndex** (`llama-index-readers-github`) | Reads repo files through the GitHub API for ingestion |
| Embeddings and reranking | `sentence-transformers`, `langchain-huggingface` | Embeddings and a cross-encoder reranker |
| Retrieval evaluation | **RAGAS** | Faithfulness scoring of generated output against retrieved context |
| Observability | **LangSmith** | Traces every tool call, subagent delegation and retrieval step |
| LLM providers | `langchain-groq`, `langchain-ollama`, `langchain-huggingface` | Pluggable model backends via LangChain |
| Cross-session memory | **Supabase** (Postgres via `asyncpg`) | Persistent memory beyond a single LangGraph thread |
| Sandbox | **Docker** (`docker` SDK) | Isolated shell execution with a bind-mounted workspace |
| API layer | **FastAPI** | HTTP and SSE surface over the harness |
| Editor integration | **VS Code extension** + **ACP** (`agent-client-protocol`) | Runs the agent inside VS Code through the extension, which talks to the ACP server |
| Browser UI | Web UI (`ui/`) | Use the agent from the browser, on top of the FastAPI backend |
| Serialization | **TOON** (`toon`) | Compact encoding to cut token usage |
| Testing and tooling | `pytest`, `ruff`, `uv` | Verification and dependency management |

Requires **Python 3.12+**.

---

## How it works

```
User request
     |
     v
+---------------------------------------------------------------+
|  CodingAgentHarness  (run / resume, thread-scoped)            |
|                                                               |
|  1. Planning subagent  ->  PLAN.md                            |
|  2. Design subagent    ->  ARCHITECTURE.md                    |
|  3. Build subagent     ->  code, commits, deploy              |
+---------------------------------------------------------------+
```

### The build loop

The build subagent works one todo at a time:

```
Orientation: read AGENTS.md, PLAN.md, ARCHITECTURE.md, progress.md, git status
        |
        v
Decompose into todos (write_todos)
        |
   +--> Retrieve context (codebase GraphRAG + external docs)
   |         |
   |         v
   |    Generate code for this todo (writes are staged for approval)
   |         |
   |         v
   |    make verify (lint, format, tests)
   |         |  pass                      | fail
   |         v                            v
   |    Commit, mark todo done      Feed the error text back, retry
   |         |                      (capped at 2-3 attempts, then escalate to a human)
   +--- more todos? --- yes
        | no
        v
End-to-end check via Chrome DevTools MCP
        |
        v
Feature complete
```

### Shared state

All three stages operate on one LangGraph `AgentState`: the user request, `plan_md`, `architecture_md`, retrieved context, staged files pending approval, todos, verification attempts and result, messages, and the current stage (`planning`, `design`, `build`, `verify`, `done`).

### Entry points

`CodingAgentHarness` exposes exactly two entry points, both scoped to a `thread_id`:

- `run(user_request, thread_id)` starts a build. It returns either a final result or an `awaiting_approval` status when a human decision is needed.
- `resume(thread_id, approved)` continues a paused run from its checkpoint.

Because both are thread-scoped, a crashed or interrupted build resumes where it stopped instead of restarting planning, design and build.

### Todo mechanism

The build subagent keeps a self-maintained task list with a single `write_todos` tool that replaces the whole list on each call. Middleware re-injects the current list into every model turn, so the objective stays in recent context. Rules: only one todo is `in_progress` at a time, todos are marked done immediately, and completion is gated on verification results rather than the model's own judgement. Details are in `TODO_MECHANISM_SPEC.md`.

---

## Harness architecture

The harness follows a five-component model. Each component maps to concrete files in this repo.

| # | Component | What it looks like here |
|---|---|---|
| 1 | **System of record** | `harness/AGENTS.md` (root instruction file), `PLAN.md`, `ARCHITECTURE.md`, the `skills/` directory, and an `AGENT.md` memory section. Subagents read these before acting instead of re-deriving scope. |
| 2 | **Tools** | Deep Agents filesystem and shell backend, GitHub MCP, Vercel/Render MCP, Tavily, Chrome DevTools MCP, a custom API toolset, and codebase/docs retrievers. |
| 3 | **Feedback and verification** | `make verify` (lint, format, test), Chrome DevTools MCP for E2E, RAGAS for retrieval faithfulness. |
| 4 | **Guardrails and permissions** | `harness/permissions.json` with `allow` / `ask` / `deny` rules, a HITL approval gate, and sandboxing of all execution. |
| 5 | **Observability and memory** | LangSmith traces, `progress.md`, `failures.md`, `write_todos` state, and Supabase for cross-session memory. |

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
      permissions.json           GitHub, Vercel, Render, Chrome DevTools,
      HITL approvals             Tavily, Neo4j, Supabase, LangSmith
      Docker sandbox
```

### Guardrails in detail

`permissions.json` defines three tiers:

- **allow**: routine, safe actions such as running `pytest` and `ruff`, and editing source files.
- **ask**: actions that pause for human approval, such as `git push`, deploys and GitHub MCP write calls. These entries become interrupts on the matching Deep Agents tools.
- **deny**: never permitted, such as reading `.env` or `secrets/**` and `rm -rf`.

Hard rules in `AGENTS.md`: never edit or delete a test to make a run pass, never push or deploy without an approval checkpoint, and never add a dependency without flagging it in `PLAN.md`. Repeated mistakes escalate: fix in session, then add a line to `AGENTS.md`, then turn it into a `deny` rule or a `make verify` check.

### Verification tiers

1. **Static**: lint and format after every edit.
2. **Unit and integration**: `pytest` before every commit.
3. **End to end**: Chrome DevTools MCP drives the built app, required before a feature is declared done.

On failure, the real stderr or traceback goes into the next generation call. There are no blind retries.

### Design rule: one framework per file

Each file under `frameworks/` wraps exactly one external framework or API behind a class. Nothing outside `frameworks/` imports a raw framework directly. This keeps the code OOP-style and makes any integration swappable.

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

For hosted, multi-tenant use, the sandbox never touches the user's real repository. It works on a disposable copy:

1. **Clone** the repo into an ephemeral directory with a short-lived credential.
2. **Execute** inside a Docker container that bind-mounts only that clone.
3. **Verify** with lint, tests and build inside the sandbox.
4. **Push** only if verification passed, using a fresh credential held by the orchestrator, never by the container.
5. **Cleanup** by deleting the clone.

Container restrictions include a writable mount only for the run directory, a network allowlist (package registries, not GitHub), CPU and memory limits, dropped Linux capabilities, and an ephemeral container per command. After any run the real repository is either unchanged or updated with a verified change, with no partial state in between. Full details in `SANDBOX_EXECUTION_LIFECYCLE.md`.

---

## Project structure

```
agent/          Orchestrator: CodingAgentHarness, shared state
frameworks/     One wrapper class per external framework (LangGraph, Deep Agents,
                MCP client, GitHub/Vercel/Render/Chrome DevTools MCP, Tavily,
                LangSmith, RAGAS, Supabase, Docker sandbox, VS Code workspace)
nodes/          Planning, design and build subagents
tools/          API toolset, deploy toolset, codebase and docs retrievers
graphrag/       Neo4j GraphRAG ingestion and retrieval
harness/        AGENTS.md, permissions.json, progress.md, failures.md
interfaces/     api_server.py (FastAPI), acp_server.py (ACP, used by the VS Code extension)
skills/         Reusable, project-specific procedures the agent can reference
ui/             Browser UI
vscode-extension/   VS Code extension (ACP client)
tests/          pytest suite
cli.py          Command-line entry point
Makefile        verify, lint, test, run-api
Dockerfile.sandbox  Sandbox image
pyproject.toml, uv.lock   Dependencies
```

---

## Getting started

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker, and a running Neo4j instance if you use GraphRAG.

```bash
git clone https://github.com/Muditshrivastav/AI-Coding-Agent.git
cd AI-Coding-Agent

# Install dependencies
uv sync

# Configure environment
cp .env.example .env     # then fill in your keys

# Build the sandbox image
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
make verify         # ruff check, ruff format, pytest
```

---

## Configuration

Copy `.env.example` to `.env`. It covers:

| Group | Variables |
|---|---|
| Search and deploy | `tavily_api_key`, `render_api_key` |
| GitHub OAuth app | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, `GITHUB_REDIRECT_URI` |
| GitHub ingestion | `GITHUB_TOKEN` |
| Supabase memory | `supabase_uri`, `supabase_url`, `supabase_anon_key`, `supabase_service_role` |
| LangSmith tracing | `LANGSMITH_*` and `LANGCHAIN_*` variables |
| Neo4j GraphRAG | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |

Never commit `.env`. Tune what the agent may do without asking in `harness/permissions.json`.

---

## Further documentation

| File | Contents |
|---|---|
| `CODING_AGENT_IMPLEMENTATION.md` | Full implementation spec: stack, harness components, subagents, build loop |
| `GRAPHRAG_DESIGN.md` | GraphRAG ingestion, retrieval, reranking and anti-hallucination design |
| `SANDBOX_EXECUTION_LIFECYCLE.md` | Clone, sandbox, verify, push and cleanup lifecycle |
| `TODO_MECHANISM_SPEC.md` | The `write_todos` task-tracking mechanism |

---

## Open items

Carried over from the implementation spec:

- Concurrency: one build at a time in v1 versus parallel threads.
- API authentication (API keys versus running behind a private network) for the FastAPI layer, since it can trigger real pushes and deploys.
- Exact Deep Agents `interrupt_before` parameter names to confirm against the installed version when wiring `permissions.json`.
- Full contents of `skills/` and the `AGENT.md` memory section.
