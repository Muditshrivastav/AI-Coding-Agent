"""
frameworks/deepagents_backend.py - Wraps deepagents LocalShellBackend / DockerSandboxBackend
with HarnessGuard HITL enforcement, and provides the DeepAgentsBackend façade used by agent/core.py.

GuardedShellBackend fully implements deepagents.backends.protocol.BackendProtocol so that
FilesystemMiddleware and SubAgentMiddleware can perform class-level introspection without
hitting 'AttributeError: type object … has no attribute …'.

Guarded methods (read, write, edit, delete, execute, upload, download) pass through the
three-gate permission pipeline before accessing the filesystem or shell. Remaining
backend-specific attributes are forwarded transparently.
"""

import logging
import os
import posixpath
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.guardrail import HarnessGuard

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.filesystem import _resolve_ripgrep_path
from deepagents.backends.protocol import (
    BackendProtocol,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
)
from deepagents.backends.utils import InvalidGlobPatternError, compile_grep_include_glob
from deepagents.middleware import filesystem as filesystem_middleware

from frameworks.docker_sandbox import DockerSandboxBackend
from frameworks.model_config import create_chat_model

logger = logging.getLogger(__name__)

# The deepagents filesystem middleware otherwise abandons glob calls after
# GLOB_TIMEOUT, even when the guarded backend is still searching.
filesystem_middleware.GLOB_TIMEOUT = None


class GuardedShellBackend(BackendProtocol):
    """Wraps LocalShellBackend or DockerSandboxBackend, enforcing HarnessGuard (HITL) on
    shell and filesystem operations.

    Inherits from ``BackendProtocol`` so that deepagents middleware can discover the
    expected interface via normal class introspection.

    Guarded operations go through the deny → ask → allow pipeline before the call
    reaches the underlying backend. Paths are normalized to workspace-relative
    form so permissions patterns work with POSIX virtual paths.
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

    @staticmethod
    def _permission_path(path: str) -> str:
        normalized = path.replace("\\", "/").lstrip("/")
        return normalized or "."

    @classmethod
    def _search_target(cls, path: str | None, pattern: str | None = None) -> str:
        normalized_path = cls._permission_path(path) if path else ""
        target = posixpath.normpath(posixpath.join(normalized_path, pattern or ""))
        return target[2:] if target.startswith("./") else target or "."

    def _enforce_path(self, action: str, path: str, description: str) -> None:
        normalized = self._permission_path(path)
        self._guard.enforce(
            f"{action}({normalized})",
            description=description,
        )

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
        self._enforce_path("Edit", file_path, f"Write file: {file_path}")
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
        self._enforce_path("Edit", file_path, f"Write file: {file_path}")
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
        self._enforce_path("Edit", file_path, f"Edit file: {file_path}")
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> Any:
        """Async edit."""
        self._enforce_path("Edit", file_path, f"Edit file: {file_path}")
        if hasattr(self._backend, "aedit"):
            return await self._backend.aedit(file_path, old_string, new_string, replace_all)
        import asyncio
        return await asyncio.to_thread(
            self._backend.edit, file_path, old_string, new_string, replace_all
        )

    def delete(self, file_path: str) -> Any:
        """Guard file/directory deletion before delegating to backend."""
        self._enforce_path("Delete", file_path, f"Delete path: {file_path}")
        return self._backend.delete(file_path)

    async def adelete(self, file_path: str) -> Any:
        """Async delete."""
        self._enforce_path("Delete", file_path, f"Delete path: {file_path}")
        if hasattr(self._backend, "adelete"):
            return await self._backend.adelete(file_path)
        import asyncio
        return await asyncio.to_thread(self._backend.delete, file_path)

    # ------------------------------------------------------------------
    # READ-ONLY / STRUCTURAL
    # ------------------------------------------------------------------

    def ls(self, path: str) -> Any:
        self._enforce_path("Read", path, f"List directory: {path}")
        return self._backend.ls(path)

    async def als(self, path: str) -> Any:
        self._enforce_path("Read", path, f"List directory: {path}")
        if hasattr(self._backend, "als"):
            return await self._backend.als(path)
        import asyncio
        return await asyncio.to_thread(self._backend.ls, path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        self._enforce_path("Read", file_path, f"Read file: {file_path}")
        return self._backend.read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
        self._enforce_path("Read", file_path, f"Read file: {file_path}")
        if hasattr(self._backend, "aread"):
            return await self._backend.aread(file_path, offset, limit)
        import asyncio
        return await asyncio.to_thread(self._backend.read, file_path, offset, limit)

    def glob(self, pattern: str, path: str | None = None) -> Any:
        target = self._search_target(path, pattern)
        self._enforce_path("Read", target, f"Find files matching {pattern}")
        if isinstance(self._backend, FilesystemBackend):
            return self._glob_filesystem(pattern, path)
        return self._backend.glob(pattern, path)

    async def aglob(self, pattern: str, path: str | None = None) -> Any:
        target = self._search_target(path, pattern)
        self._enforce_path("Read", target, f"Find files matching {pattern}")
        if isinstance(self._backend, FilesystemBackend):
            import asyncio

            return await asyncio.to_thread(self._glob_filesystem, pattern, path)
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
        target = self._search_target(path, glob)
        self._enforce_path("Read", target, f"Search files under {target}")
        if isinstance(self._backend, FilesystemBackend):
            return self._grep_filesystem(pattern, path, glob, max_count=max_count)
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
        target = self._search_target(path, glob)
        self._enforce_path("Read", target, f"Search files under {target}")
        if isinstance(self._backend, FilesystemBackend):
            import asyncio

            return await asyncio.to_thread(
                self._grep_filesystem, pattern, path, glob, max_count=max_count
            )
        kwargs: dict[str, Any] = {}
        if glob is not None:
            kwargs["glob"] = glob
        if max_count is not None:
            kwargs["max_count"] = max_count
        if hasattr(self._backend, "agrep"):
            return await self._backend.agrep(pattern, path, **kwargs)
        import asyncio
        return await asyncio.to_thread(self._backend.grep, pattern, path, **kwargs)

    def _glob_filesystem(self, pattern: str, path: str | None) -> GlobResult:
        """Walk a local filesystem without the library's fixed glob deadline."""
        try:
            matches_pattern = compile_grep_include_glob(pattern)
        except InvalidGlobPatternError as exc:
            return GlobResult(error=str(exc), matches=None)

        try:
            search_path = (
                self._backend.cwd
                if path is None or path == "/"
                else self._backend._resolve_path(path)
            )
            if not search_path.exists() or not search_path.is_dir():
                return GlobResult(matches=[])
        except (OSError, RuntimeError) as exc:
            display_path = path if path is not None else "<default>"
            return GlobResult(error=f"Error globbing path '{display_path}': {exc}", matches=[])

        results: list[FileInfo] = []
        try:
            for matched_path in search_path.rglob("*"):
                try:
                    rel_path = matched_path.relative_to(search_path).as_posix()
                    if not matches_pattern(rel_path) or not matched_path.is_file():
                        continue
                except (PermissionError, OSError, RuntimeError, ValueError):
                    continue

                if self._backend.virtual_mode:
                    try:
                        matched_path.resolve().relative_to(self._backend.cwd)
                        result_path = self._backend._to_virtual_path(matched_path)
                    except (OSError, RuntimeError, ValueError):
                        continue
                else:
                    result_path = str(matched_path)

                try:
                    stat = matched_path.stat()
                    results.append(
                        {
                            "path": result_path,
                            "is_dir": False,
                            "size": int(stat.st_size),
                            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )
                except OSError:
                    results.append({"path": result_path, "is_dir": False})
        except (OSError, RuntimeError, ValueError) as exc:
            display_path = path if path is not None else "<default>"
            message = f"Glob of '{display_path}' aborted partway: {exc}"
            logger.warning("%s", message, exc_info=True)
            results.sort(key=lambda result: result["path"])
            return GlobResult(error=message, matches=results)

        results.sort(key=lambda result: result["path"])
        return GlobResult(matches=results)

    def _grep_filesystem(
        self,
        pattern: str,
        path: str | None,
        include_glob: str | None,
        *,
        max_count: int | None,
    ) -> GrepResult:
        """Search a local filesystem without the library's fixed grep deadline."""
        refusal = self._backend._refused_grep_glob_error(include_glob)
        if refusal is not None:
            return GrepResult(error=refusal, matches=[])

        try:
            base_path = self._backend._resolve_path(path or ".")
        except ValueError:
            return GrepResult(matches=[])
        except (OSError, RuntimeError) as exc:
            return GrepResult(error=f"Error searching path '{path or '.'}': {exc}", matches=[])

        try:
            if not base_path.exists():
                return GrepResult(matches=[])
        except OSError as exc:
            return GrepResult(error=f"Error searching path '{path or '.'}': {exc}", matches=[])

        matcher = compile_grep_include_glob(include_glob) if include_glob else None
        root = base_path if base_path.is_dir() else base_path.parent
        rg_results = self._ripgrep_filesystem(
            pattern, base_path, include_glob, max_count=max_count
        )
        if rg_results is not None:
            results, truncated = rg_results
            matches = [
                {"path": result_path, "line": line, "text": text}
                for result_path, lines in results.items()
                for line, text in lines
            ]
            return GrepResult(matches=matches, truncated=truncated)

        matches: list[GrepMatch] = []
        file_errors: list[str] = []

        try:
            for file_path in root.rglob("*"):
                try:
                    if not file_path.is_file():
                        continue
                    if matcher is not None and not matcher(file_path.relative_to(root).as_posix()):
                        continue
                    if file_path.stat().st_size > self._backend.max_file_size_bytes:
                        continue
                    if self._backend.virtual_mode:
                        try:
                            virtual_path = self._backend._to_virtual_path(file_path)
                        except ValueError:
                            continue
                        except (OSError, RuntimeError):
                            logger.warning("Could not resolve grep result path.", exc_info=True)
                            continue
                    else:
                        virtual_path = str(file_path)

                    line_count = 0
                    with file_path.open(encoding="utf-8", errors="strict") as source:
                        for line_number, raw_line in enumerate(source, 1):
                            line_count = line_number
                            if pattern not in raw_line:
                                continue
                            if max_count is not None and len(matches) >= max_count:
                                return GrepResult(
                                    error=self._format_grep_file_errors(file_errors),
                                    matches=matches,
                                    truncated=True,
                                )
                            matches.append(
                                {
                                    "path": virtual_path,
                                    "line": line_number,
                                    "text": raw_line.rstrip("\n"),
                                }
                            )
                except UnicodeDecodeError as exc:
                    if line_count:
                        file_errors.append(f"- {virtual_path}: {type(exc).__name__}: {exc.reason}")
                except (OSError, RuntimeError) as exc:
                    file_errors.append(f"- {file_path.name}: {type(exc).__name__}")
        except (OSError, RuntimeError, ValueError) as exc:
            return GrepResult(
                error=f"Search aborted partway: {type(exc).__name__}",
                matches=matches,
            )

        return GrepResult(
            error=self._format_grep_file_errors(file_errors),
            matches=matches,
        )

    def _ripgrep_filesystem(
        self,
        pattern: str,
        base_path: Path,
        include_glob: str | None,
        *,
        max_count: int | None,
    ) -> tuple[dict[str, list[tuple[int, str]]], bool] | None:
        """Use ripgrep without starting the library's fixed-duration watchdog."""
        rg_path = _resolve_ripgrep_path()
        if rg_path is None:
            return None

        command = [rg_path, "--json", "-F"]
        if max_count is not None:
            command.extend(["-m", str(max_count + 1)])
        if include_glob:
            command.extend(["--glob", include_glob])

        search_cwd = None
        if base_path.is_dir():
            command.extend(["--", pattern, "."])
            search_cwd = str(base_path)
        else:
            command.extend(["--", pattern, str(base_path)])

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=search_cwd,
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            _resolve_ripgrep_path.cache_clear()
            return None

        assert process.stdout is not None
        results: dict[str, list[tuple[int, str]]] = {}
        base_resolved = base_path.resolve()
        total = 0
        truncated = False
        try:
            for output_line in process.stdout:
                match = self._backend._parse_rg_match(
                    output_line,
                    base_path,
                    base_resolved,
                )
                if match is None:
                    continue
                result_path, line, text = match
                if max_count is not None and total >= max_count:
                    truncated = True
                    process.terminate()
                    break
                results.setdefault(result_path, []).append((line, text.rstrip("\r\n")))
                total += 1
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()

        return_code = process.wait()
        if return_code not in (0, 1) and not truncated:
            logger.warning("ripgrep exited with code %s; using the Python search fallback.", return_code)
            return None
        return results, truncated

    @staticmethod
    def _format_grep_file_errors(file_errors: list[str]) -> str | None:
        if not file_errors:
            return None
        return "One or more files could not be fully searched:\n" + "\n".join(file_errors)

    def upload_files(self, files: list[tuple[str, bytes]]) -> Any:
        for path, _ in files:
            self._enforce_path("Edit", path, f"Upload file: {path}")
        return self._backend.upload_files(files)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> Any:
        for path, _ in files:
            self._enforce_path("Edit", path, f"Upload file: {path}")
        if hasattr(self._backend, "aupload_files"):
            return await self._backend.aupload_files(files)
        import asyncio
        return await asyncio.to_thread(self._backend.upload_files, files)

    def download_files(self, paths: list[str]) -> Any:
        for path in paths:
            self._enforce_path("Read", path, f"Download file: {path}")
        return self._backend.download_files(paths)

    async def adownload_files(self, paths: list[str]) -> Any:
        for path in paths:
            self._enforce_path("Read", path, f"Download file: {path}")
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
        sandbox_mode: str | None = None,
        docker_image: str = "ai-coding-agent-sandbox:latest",
    ) -> None:
        self._root_dir = root_dir
        self._guard = guard or HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
        self._sandbox_mode = (
            sandbox_mode or os.environ.get("SANDBOX_MODE", "auto")
        ).lower()
        if self._sandbox_mode not in {"auto", "docker", "local"}:
            raise ValueError("SANDBOX_MODE must be 'auto', 'docker', or 'local'.")

        if self._sandbox_mode == "local":
            raw_backend: Any = LocalShellBackend(
                root_dir=root_dir,
                timeout=900,
                inherit_env=True,
            )
            logger.info("Sandbox active mode: local host shell.")
        else:
            raw_backend = DockerSandboxBackend(
                root_dir=root_dir,
                image=docker_image,
                auto_fallback=(self._sandbox_mode == "auto"),
            )
            docker_available = raw_backend.is_docker_available
            if docker_available:
                logger.info("Sandbox active mode: Docker (%s).", self._sandbox_mode)
            elif self._sandbox_mode == "auto":
                logger.warning(
                    "Sandbox active mode: local fallback (Docker is unavailable)."
                )
            else:
                logger.warning("Docker sandbox requested but Docker is unavailable.")

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

    @backend.setter
    def backend(self, backend: GuardedShellBackend | Any) -> None:
        """Update backend. If not already wrapped in GuardedShellBackend, wrap it."""
        if isinstance(backend, GuardedShellBackend):
            self._backend = backend
            self._guard = backend._guard
        else:
            self._backend = GuardedShellBackend(backend, self._guard)

    @property
    def raw_backend(self) -> Any:
        """Return the underlying un-guarded backend instance."""
        return self._backend.raw_backend

    @raw_backend.setter
    def raw_backend(self, raw: Any) -> None:
        """Set a new underlying backend and rebuild GuardedShellBackend wrapper."""
        self._backend = GuardedShellBackend(raw, self._guard)

    @property
    def guard(self) -> HarnessGuard:
        """Return the active HarnessGuard instance."""
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
        import inspect

        from deepagents import create_deep_agent

        resolved_model: Any

        if isinstance(model, str) and model.startswith("ollama:"):
            resolved_model = create_chat_model(model, temperature=0.2, num_ctx=8192)
        elif isinstance(model, str):
            resolved_model = create_chat_model(model, temperature=0.2)
        else:
            resolved_model = model

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "backend": self._backend,
            "checkpointer": checkpointer,
        }

        # Only pass subagents if the installed deepagents version accepts it.
        sig = inspect.signature(create_deep_agent)
        if "subagents" in sig.parameters and subagents:
            kwargs["subagents"] = subagents

        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if tools:
            kwargs["tools"] = tools
        if state_schema:
            kwargs["state_schema"] = state_schema

        return create_deep_agent(**kwargs)
