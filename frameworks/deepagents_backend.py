"""
frameworks/deepagents_backend.py - Wraps deepagents LocalShellBackend / DockerSandboxBackend
with HarnessGuard HITL enforcement, and provides the DeepAgentsBackend façade used by agent/core.py.

GuardedShellBackend fully implements deepagents.backends.protocol.BackendProtocol so that
FilesystemMiddleware and SubAgentMiddleware can perform class-level introspection without
hitting 'AttributeError: type object … has no attribute …'.

Guarded methods (write, edit, delete, execute) pass through the three-gate HITL pipeline
before touching the filesystem or shell.  All other protocol methods are forwarded
transparently to the underlying backend instance.
"""

import os
from typing import Any

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from agent.guardrail import HarnessGuard

from frameworks.docker_sandbox import DockerSandboxBackend


class GuardedShellBackend(BackendProtocol):
    """Wraps LocalShellBackend or DockerSandboxBackend, enforcing HarnessGuard (HITL) on
    all shell and mutating filesystem operations.

    Inherits from ``BackendProtocol`` so that deepagents middleware (FilesystemMiddleware,
    SubAgentMiddleware, etc.) can discover the full interface via normal class introspection
    without triggering ``AttributeError`` on the *type* object.

    Guarded operations (execute, write, edit, delete) go through the three-gate
    deny → ask → allow pipeline before the call reaches the underlying backend.
    Read-only or structural operations (ls, read, glob, grep, upload_files,
    download_files) are forwarded directly — HarnessGuard is still available for
    callers that want to add read-guards at a higher level.
    """

    def __init__(self, backend: Any, guard: HarnessGuard) -> None:
        self._backend = backend
        self._guard = guard

    # ------------------------------------------------------------------
    # Convenience accessor
    # ------------------------------------------------------------------

    @property
    def raw_backend(self) -> Any:
        return self._backend

    # ------------------------------------------------------------------
    # GUARDED — mutating / shell operations
    # ------------------------------------------------------------------

    def execute(self, command: str) -> Any:
        """Execute a shell command after HITL guard check (deny → ask → allow)."""
        cmd_stripped = command.strip()
        self._guard.enforce(
            f"Bash({cmd_stripped})",
            description=f"Execute: {cmd_stripped}",
        )
        return self._backend.execute(cmd_stripped)

    async def aexecute(self, command: str) -> Any:
        """Async execute — guard is synchronous (interrupt blocks), result is awaited."""
        cmd_stripped = command.strip()
        self._guard.enforce(
            f"Bash({cmd_stripped})",
            description=f"Execute: {cmd_stripped}",
        )
        if hasattr(self._backend, "aexecute"):
            return await self._backend.aexecute(cmd_stripped)
        import asyncio
        return await asyncio.to_thread(self._backend.execute, cmd_stripped)

    def write(self, file_path: str, content: str) -> Any:
        """Guard file writes (Edit/Write pattern) before delegating to backend."""
        self._guard.enforce(
            f"Edit({file_path})",
            description=f"Write file: {file_path}",
        )
        if hasattr(self._backend, "write"):
            return self._backend.write(file_path, content)
        # Graceful fallback for backends that expose write_file instead
        if hasattr(self._backend, "write_file"):
            return self._backend.write_file(file_path, content)
        full_path = (
            os.path.join(self._backend.root_dir, file_path.lstrip("/"))
            if hasattr(self._backend, "root_dir")
            else file_path
        )
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"Wrote {len(content)} chars to {file_path}"

    async def awrite(self, file_path: str, content: str) -> Any:
        """Async write — guard is synchronous, I/O is async."""
        self._guard.enforce(
            f"Edit({file_path})",
            description=f"Write file: {file_path}",
        )
        if hasattr(self._backend, "awrite"):
            return await self._backend.awrite(file_path, content)
        import asyncio
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> Any:
        """Guard file edits before delegating to backend."""
        self._guard.enforce(
            f"Edit({file_path})",
            description=f"Edit file: {file_path}",
        )
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> Any:
        """Async edit."""
        self._guard.enforce(
            f"Edit({file_path})",
            description=f"Edit file: {file_path}",
        )
        if hasattr(self._backend, "aedit"):
            return await self._backend.aedit(file_path, old_string, new_string, replace_all)
        import asyncio
        return await asyncio.to_thread(
            self._backend.edit, file_path, old_string, new_string, replace_all
        )

    def delete(self, file_path: str) -> Any:
        """Guard file/directory deletion before delegating to backend."""
        self._guard.enforce(
            f"Delete({file_path})",
            description=f"Delete path: {file_path}",
        )
        return self._backend.delete(file_path)

    async def adelete(self, file_path: str) -> Any:
        """Async delete."""
        self._guard.enforce(
            f"Delete({file_path})",
            description=f"Delete path: {file_path}",
        )
        if hasattr(self._backend, "adelete"):
            return await self._backend.adelete(file_path)
        import asyncio
        return await asyncio.to_thread(self._backend.delete, file_path)

    # ------------------------------------------------------------------
    # READ-ONLY / STRUCTURAL — forwarded directly (no guard gate)
    # ------------------------------------------------------------------

    def ls(self, path: str) -> Any:
        return self._backend.ls(path)

    async def als(self, path: str) -> Any:
        if hasattr(self._backend, "als"):
            return await self._backend.als(path)
        import asyncio
        return await asyncio.to_thread(self._backend.ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        return self._backend.read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        if hasattr(self._backend, "aread"):
            return await self._backend.aread(file_path, offset, limit)
        import asyncio
        return await asyncio.to_thread(self._backend.read, file_path, offset, limit)

    def glob(self, pattern: str, path: str | None = None) -> Any:
        return self._backend.glob(pattern, path)

    async def aglob(self, pattern: str, path: str | None = None) -> Any:
        if hasattr(self._backend, "aglob"):
            return await self._backend.aglob(pattern, path)
        import asyncio
        return await asyncio.to_thread(self._backend.glob, pattern, path)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if glob is not None:
            kwargs["glob"] = glob
        if max_count is not None:
            kwargs["max_count"] = max_count
        return self._backend.grep(pattern, path, **kwargs)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if glob is not None:
            kwargs["glob"] = glob
        if max_count is not None:
            kwargs["max_count"] = max_count
        if hasattr(self._backend, "agrep"):
            return await self._backend.agrep(pattern, path, **kwargs)
        import asyncio
        return await asyncio.to_thread(self._backend.grep, pattern, path, **kwargs)

    def upload_files(self, files: list[tuple[str, bytes]]) -> Any:
        return self._backend.upload_files(files)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> Any:
        if hasattr(self._backend, "aupload_files"):
            return await self._backend.aupload_files(files)
        import asyncio
        return await asyncio.to_thread(self._backend.upload_files, files)

    def download_files(self, paths: list[str]) -> Any:
        return self._backend.download_files(paths)

    async def adownload_files(self, paths: list[str]) -> Any:
        if hasattr(self._backend, "adownload_files"):
            return await self._backend.adownload_files(paths)
        import asyncio
        return await asyncio.to_thread(self._backend.download_files, paths)

    # ------------------------------------------------------------------
    # Legacy compatibility shims (used by older harness code paths)
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> Any:
        """Legacy alias for write() kept for backward compatibility."""
        return self.write(path, content)

    def read_file(self, path: str) -> Any:
        """Legacy alias for read() kept for backward compatibility."""
        return self.read(path)

    # ------------------------------------------------------------------
    # Transparent attribute proxy for any backend-specific extras
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute not defined above to the underlying backend instance."""
        return getattr(self._backend, name)


class DeepAgentsBackend:
    """Wraps the deepagents library, providing a root_dir-scoped backend with active
    HarnessGuard HITL.

    Supports running in 'auto' (Docker sandbox with local fallback), 'docker' (strictly
    Docker), or 'local' (host shell) modes.
    """

    def __init__(
        self,
        root_dir: str = ".",
        guard: HarnessGuard | None = None,
        sandbox_mode: str = "auto",
        docker_image: str = "ai-coding-agent-sandbox:latest",
    ) -> None:
        self._root_dir = root_dir
        self._guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
        self._sandbox_mode = sandbox_mode

        if sandbox_mode == "local":
            raw_backend: Any = LocalShellBackend(root_dir=root_dir)
        else:
            # "auto" or "docker"
            raw_backend = DockerSandboxBackend(
                root_dir=root_dir,
                image=docker_image,
                auto_fallback=(sandbox_mode == "auto"),
            )

        self._backend = GuardedShellBackend(raw_backend, self._guard)

    @property
    def sandbox_mode(self) -> str:
        return self._sandbox_mode

    @property
    def is_docker_active(self) -> bool:
        raw = self.raw_backend
        if isinstance(raw, DockerSandboxBackend):
            return raw.is_docker_available
        return False

    @property
    def backend(self) -> GuardedShellBackend:
        return self._backend

    @property
    def raw_backend(self) -> Any:
        return self._backend._backend

    @property
    def guard(self) -> HarnessGuard:
        return self._guard

    def execute_command(self, command: str) -> Any:
        """Directly execute a command guarded by HarnessGuard HITL pipeline."""
        return self._backend.execute(command)

    def build_agent(
        self,
        model: Any,
        subagents: list[Any],
        system_prompt: str | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
        state_schema: Any = None,
    ) -> Any:
        from deepagents import create_deep_agent

        # If a string provider model was given for groq, configure it with ChatOllama fallback
        resolved_model = model
        if isinstance(model, str) and model.startswith("groq:"):
            from langchain_groq import ChatGroq
            from langchain_ollama import ChatOllama
            bare_model = model[len("groq:"):]
            primary = ChatGroq(model=bare_model, temperature=0.2)
            fallback = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)
            resolved_model = primary.with_fallbacks([fallback])

        kwargs: dict[str, Any] = {
            "model": resolved_model,
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
