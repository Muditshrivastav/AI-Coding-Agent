# Project: Coding Agent Harness
An autonomous coding agent built as an execution harness to plan, design, build, verify, and deploy software projects end-to-end.

## Structure
- `agent/`: Orchestrator harness and shared state.
- `nodes/`: Planning, design, and build subagents.
- `frameworks/`: Isolated class wrappers (strictly one external framework per file).
- `tools/`: Custom API-backed tools, retrievers, and deploy tools.
- `interfaces/`: Streamlit HITL approval UI, FastAPI server, and ACP server.
- `harness/`: System of record and operational guardrails.

## Commands
- Verify (all checks): `make verify`
- Lint: `ruff check . --fix`
- Format: `ruff format .`
- Test: `pytest -q`

## Conventions
1. Each file under `frameworks/` wraps exactly one framework — never mix raw external library calls across files.
2. Subagents read `PLAN.md` / `ARCHITECTURE.md` via file reading tools — never re-derive scope themselves.
3. All shell/file-write actions go through `LocalShellBackend` — nothing executes outside `root_dir`.
4. Staged writes must be approved before execution.

## Do Not
- Edit or delete a test to make a run pass.
- Push to GitHub or trigger a deploy without an approval checkpoint.
- Add a new dependency without flagging it in `PLAN.md` first.
