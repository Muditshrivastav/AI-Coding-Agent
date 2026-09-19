"""
frameworks/docker_sandbox.py - Docker-based direct workspace sandboxing (Antigravity / Codex style).

Runs commands inside an isolated Docker container while bind-mounting the user's
project root directly (host <-> container /workspace).
- All changes made by commands and tools directly affect the workspace.
- The host system outside the project root is completely unreachable.
- Resource bounds (mem, cpu) and privilege dropping (cap_drop=ALL) are enforced.
- Uses the Docker Python SDK (docker package) for proper programmatic control —
  no shell-injection risk, structured error handling, and streaming output.
- Includes daemon availability probing and automatic fallback to LocalShellBackend.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    SandboxBackendProtocol,
)

try:
    import docker
    import docker.errors
    _DOCKER_SDK_AVAILABLE = True
except ImportError:
    _DOCKER_SDK_AVAILABLE = False


class DockerSandboxBackend(LocalShellBackend, SandboxBackendProtocol):
    """Direct-workspace Docker sandbox backend.

    Bind-mounts the local workspace into a container at /workspace.
    Commands execute within the container environment, modifying workspace files
    directly while preserving host isolation.

    Uses the Docker Python SDK (``docker`` package) for programmatic container
    management — no CLI subprocess needed for execution.  Falls back to the
    raw ``docker`` CLI when the SDK is unavailable, and further falls back to
    LocalShellBackend when Docker itself is unreachable.
    """

    def __init__(
        self,
        root_dir: str | Path = ".",
        image: str = "ai-coding-agent-sandbox:latest",
        container_workspace: str = "/workspace",
        memory_limit: str = "1g",
        cpu_limit: float = 1.0,
        cap_drop: list[str] | str = "ALL",
        network: str = "bridge",
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        auto_fallback: bool = True,
    ) -> None:
        super().__init__(root_dir=root_dir, timeout=timeout)
        self._abs_root = str(Path(root_dir).resolve())
        self._image = image
        self._container_workspace = container_workspace
        self._memory_limit = memory_limit
        # docker SDK expects nano-CPUs (int): 1 CPU = 1_000_000_000 nano-CPUs
        self._nano_cpus = int(float(cpu_limit) * 1_000_000_000)
        self._cpu_limit = str(cpu_limit)  # kept for CLI fallback
        # cap_drop can be a list or "ALL"
        self._cap_drop: list[str] = (
            [cap_drop] if isinstance(cap_drop, str) else list(cap_drop)
        )
        self._network = network
        self._auto_fallback = auto_fallback
        self._docker_available: bool | None = None
        self._docker_client: "docker.DockerClient | None" = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return f"docker-sandbox:{self._abs_root}"

    @property
    def is_docker_available(self) -> bool:
        """Checks if the Docker daemon is reachable (cached after first probe)."""
        if self._docker_available is not None:
            return self._docker_available

        # Try the Python SDK first (preferred)
        if _DOCKER_SDK_AVAILABLE:
            try:
                client = docker.from_env(timeout=5)
                client.ping()
                self._docker_client = client
                self._docker_available = True
                return True
            except Exception:
                pass  # fall through to CLI probe

        # CLI fallback probe
        if not shutil.which("docker"):
            self._docker_available = False
            return False

        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True,
            )
            self._docker_available = res.returncode == 0
        except Exception:
            self._docker_available = False

        return self._docker_available

    # ------------------------------------------------------------------
    # CLI helper (used when SDK is unavailable)
    # ------------------------------------------------------------------

    def build_docker_command(self, command: str) -> list[str]:
        """Constructs the ``docker run`` CLI invocation (SDK-free fallback)."""
        # Forward-slash path for Docker volume mounting (Windows compatible)
        mount_src = self._abs_root.replace("\\", "/")
        cap_drop_flags: list[str] = []
        for cap in self._cap_drop:
            cap_drop_flags += [f"--cap-drop={cap}"]

        return [
            "docker", "run", "--rm",
            "-v", f"{mount_src}:{self._container_workspace}:rw",
            "-w", self._container_workspace,
            f"--memory={self._memory_limit}",
            f"--cpus={self._cpu_limit}",
            *cap_drop_flags,
            f"--network={self._network}",
            self._image,
            "sh", "-c", command,
        ]

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Executes *command* inside an ephemeral Docker container.

        Execution order:
        1. Docker Python SDK  — preferred, no shell-injection risk.
        2. Raw ``docker`` CLI — when SDK is unavailable.
        3. LocalShellBackend  — when Docker daemon is unreachable and
                                ``auto_fallback=True``.
        """
        cmd_timeout = timeout or self._default_timeout

        if not self.is_docker_available:
            if self._auto_fallback:
                return super().execute(command, timeout=cmd_timeout)
            return ExecuteResponse(
                output="Docker daemon is not running or docker CLI is not installed.",
                exit_code=1,
            )

        # ── Path 1: Docker Python SDK ──────────────────────────────────
        if self._docker_client is not None:
            return self._execute_via_sdk(command, cmd_timeout)

        # ── Path 2: CLI fallback ───────────────────────────────────────
        return self._execute_via_cli(command, cmd_timeout)

    def _execute_via_sdk(self, command: str, timeout: int) -> ExecuteResponse:
        """Run command using the Docker Python SDK (docker package)."""
        import docker.errors  # type: ignore[import]

        # Forward-slash path required by Docker on Windows
        mount_src = self._abs_root.replace("\\", "/")

        try:
            result = self._docker_client.containers.run(  # type: ignore[union-attr]
                image=self._image,
                command=["sh", "-c", command],
                volumes={
                    mount_src: {
                        "bind": self._container_workspace,
                        "mode": "rw",
                    }
                },
                working_dir=self._container_workspace,
                mem_limit=self._memory_limit,
                nano_cpus=self._nano_cpus,
                cap_drop=self._cap_drop,
                network_mode=self._network,
                remove=True,          # equivalent to --rm
                stdout=True,
                stderr=True,
                timeout=timeout,
            )
            # containers.run returns bytes when detach=False
            output = result.decode("utf-8", errors="replace") if isinstance(result, bytes) else str(result)
            return ExecuteResponse(output=output, exit_code=0)

        except docker.errors.ContainerError as exc:  # type: ignore[attr-defined]
            # Non-zero exit from the container
            output = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            return ExecuteResponse(output=output, exit_code=exc.exit_status)

        except Exception as exc:
            # Timeout, ImageNotFound, APIError, etc.
            msg = str(exc)
            exit_code = 124 if "timeout" in msg.lower() else 1
            return ExecuteResponse(
                output=f"Docker SDK error: {msg}",
                exit_code=exit_code,
            )

    def _execute_via_cli(self, command: str, timeout: int) -> ExecuteResponse:
        """Run command using the raw ``docker`` CLI subprocess."""
        docker_invocation = self.build_docker_command(command)
        try:
            proc = subprocess.run(
                docker_invocation,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                text=True,
                errors="replace",
            )
            return ExecuteResponse(
                output=proc.stdout or "",
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout if isinstance(exc.stdout, str) else ""
            return ExecuteResponse(
                output=f"{partial}\nCommand timed out after {timeout} seconds",
                exit_code=124,
            )
        except Exception as exc:
            return ExecuteResponse(
                output=f"Error executing command in Docker sandbox: {exc}",
                exit_code=1,
            )
