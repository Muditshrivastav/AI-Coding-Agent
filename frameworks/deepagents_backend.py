import os
from typing import Any
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from agent.harness import HarnessGuard


class GuardedShellBackend:
    """Wraps LocalShellBackend to enforce HarnessGuard (HITL) on all shell and file operations."""

    def __init__(self, backend: LocalShellBackend, guard: HarnessGuard) -> None:
        self._backend = backend
        self._guard = guard

    def execute(self, command: str) -> Any:
        """Executes a shell command after running it through the 3-gate HITL guard."""
        cmd_stripped = command.strip()
        signature = f"Bash({cmd_stripped})"
        # Gate 1 (deny) -> Gate 2 (ask / interrupt) -> Gate 3 (allow)
        self._guard.enforce(signature, description=f"Execute: {cmd_stripped}")
        return self._backend.execute(cmd_stripped)

    def write_file(self, path: str, content: str) -> Any:
        """Guards file writes through HarnessGuard with Edit/Write pattern."""
        signature = f"Edit({path})"
        self._guard.enforce(signature, description=f"Write file: {path}")
        if hasattr(self._backend, "write_file"):
            return self._backend.write_file(path, content)
        # Fallback to direct file write if backend does not expose write_file
        full_path = os.path.join(self._backend.root_dir, path) if hasattr(self._backend, "root_dir") else path
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Wrote {len(content)} chars to {path}"

    def read_file(self, path: str) -> Any:
        """Guards file reads through HarnessGuard (e.g. Read(.env), Read(secrets/**))."""
        signature = f"Read({path})"
        self._guard.enforce(signature, description=f"Read file: {path}")
        if hasattr(self._backend, "read_file"):
            return self._backend.read_file(path)
        full_path = os.path.join(self._backend.root_dir, path) if hasattr(self._backend, "root_dir") else path
        with open(full_path, "r", encoding="utf-8") as fh:
            return fh.read()

    def __getattr__(self, name: str) -> Any:
        """Forward any other attributes directly to the underlying LocalShellBackend."""
        return getattr(self._backend, name)


class DeepAgentsBackend:
    """Wraps Deep Agents library, providing a root_dir scoped backend with active HarnessGuard HITL."""

    def __init__(self, root_dir: str = ".", guard: HarnessGuard | None = None) -> None:
        self._root_dir = root_dir
        self._guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
        raw_backend = LocalShellBackend(root_dir=root_dir)
        self._backend = GuardedShellBackend(raw_backend, self._guard)

    @property
    def backend(self) -> GuardedShellBackend:
        return self._backend

    @property
    def raw_backend(self) -> LocalShellBackend:
        return self._backend._backend

    @property
    def guard(self) -> HarnessGuard:
        return self._guard

    def execute_command(self, command: str) -> Any:
        """Directly executes a command guarded by HarnessGuard HITL pipeline."""
        return self._backend.execute(command)

    def build_agent(
        self,
        model: str,
        subagents: list[Any],
        system_prompt: str | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
        state_schema: Any = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "backend": self._backend,
            "subagents": subagents,
            "checkpointer": checkpointer,
        }
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if tools:
            kwargs["tools"] = tools
        if state_schema:
            kwargs["state_schema"] = state_schema

        return create_deep_agent(**kwargs)
