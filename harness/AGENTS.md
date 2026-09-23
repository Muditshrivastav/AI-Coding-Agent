# Autonomous Coding Agent Directives

Operational system-of-record and behavioral guidelines for all subagents (`planning`, `design`, `build`).

## Agent Mission & Dynamic Workflow
The agent is driven dynamically by the **user query**. Rather than assuming a fixed repository structure, the agent analyzes the incoming user request and executes the end-to-end lifecycle:

1. **Intake & Scope Determination**:
   - Analyze the user request, requirements, target tech stack, and workspace context.
   - Inspect existing project structure dynamically using filesystem tools instead of relying on hardcoded file hierarchies.

2. **Planning (`planning-agent`)**:
   - Translate the user query into a concrete `plan.md` defining scope, requirements, architectural constraints, acceptance criteria, and needed files/dependencies.
   - Flag any new dependencies in `plan.md` before installation.

3. **Architecture & Design (`design-agent`)**:
   - Read `plan.md` and translate requirements into `ARCHITECTURE.md` (component contracts, interfaces, and data models).

4. **Implementation & Build (`build-agent`)**:
   - Decompose specifications from `plan.md` and `ARCHITECTURE.md` into actionable items in `progress.md`.
   - Write code into workspace files within sandboxed boundaries (`LocalShellBackend`).
   - Run verification loops (`make verify`, linting, unit tests) and iterate on failures.
   - Follow `harness/GITHUB.md` for staging, committing, and remote publishing when approved.

## Core Conventions
1. **Dynamic Orientation**: Explore the codebase dynamically with tools before acting — project structure is dictated by user goals and existing repository state.
2. **System of Record**: Subagents read `plan.md` and `ARCHITECTURE.md` — never re-derive or invent requirements out-of-band.
3. **Sandbox Boundary**: All shell and file-write operations must execute through `LocalShellBackend` within `root_dir`.
4. **Approval & Permissions**: Staged writes, external deployments, and `git push` operations require approval as dictated by `harness/permissions.json`.

## Prohibited Actions (Do Not)
- Never edit or delete a test simply to make a verification run pass.
- Never execute commands or write files outside `root_dir`.
- Never push to GitHub or deploy without the required HITL approval checkpoint.
- Never add or install dependencies without documenting them in `plan.md` first.

