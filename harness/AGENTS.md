# Coding Agent Harness Rules

These are permanent repository conventions. Request-specific scope belongs in
`harness/TASK.md`; do not replace or edit this file for an individual task.

## Repository conventions

- Follow the established patterns, naming, formatting, and type annotations.
- Keep each framework integration in its own module under `frameworks/`.
- Make focused changes that implement the requested behavior without unrelated
  refactors.
- Do not add a dependency unless it is required, is declared in `pyproject.toml`,
  and is reflected in the lock file.
- Never edit or remove a test just to make a failing run pass. Fix the cause and
  add regression coverage for behavior changes.
- Keep credentials, generated run state, and user data out of source control.

## Task workflow

1. Read this file, `harness/TASK.md`, and the relevant source and tests.
2. For multi-step work, use root `PLAN.md` to record the implementation and
   acceptance criteria. Use `ARCHITECTURE.md` only when a design artifact is
   useful to the task.
3. Implement a coherent change and add or update focused tests.
4. Run `make check` (or its constituent Ruff and pytest commands) before
   reporting completion. Report any pre-existing or environment-related failure
   separately.
5. Update `harness/progress.md` with the current stage and task status when the
   autonomous task loop is active.

## Safety and approvals

- Respect `harness/permissions.json`; do not bypass `HarnessGuard`.
- Never push, deploy, delete user data, or perform destructive Git operations
  without explicit approval.
- Do not expose secrets or read credential files unless the task specifically
  requires it and the permission policy allows it.
- Surface command failures and verification results; never claim checks passed
  without running them.

## Runtime files

- `harness/TASK.md` is generated for the current request and may be replaced by
  the next request.
- `harness/progress.md`, `harness/failures.md`, `harness/tasks.json`, and
  `harness/sessions/` contain runtime state, not permanent documentation.
- Use `harness/failures.md.template` and `harness/progress.md.template` as the
  clean starting formats for runtime logs.
