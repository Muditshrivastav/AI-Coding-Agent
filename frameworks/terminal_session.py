"""Workspace-scoped persistent terminal tools for coding agents."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.guardrail import HarnessGuard


class TerminalCommandArgs(BaseModel):
    command: str = Field(description="Shell command to run in the persistent workspace terminal.")
    timeout_seconds: int = Field(default=30, ge=1, le=1800)


class TerminalReadArgs(BaseModel):
    wait_seconds: int = Field(default=2, ge=0, le=30)


class TerminalSessionManager:
    """Owns one persistent shell process rooted in the workspace directory."""

    def __init__(self, root_dir: str, guard: HarnessGuard) -> None:
        self._root_dir = str(Path(root_dir).resolve())
        self._guard = guard
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._output_queue: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._pending_marker: str | None = None
        self._pending_output: list[str] = []
        self._session_id = str(uuid.uuid4())
        self._platform = "powershell" if os.name == "nt" else "bash"

    def get_tools(self) -> list[StructuredTool]:
        def run_command(command: str, timeout_seconds: int = 30) -> str:
            return self.run(command, timeout_seconds=timeout_seconds)

        def read_output(wait_seconds: int = 2) -> str:
            return self.read(wait_seconds=wait_seconds)

        def close_terminal() -> str:
            self.close()
            return "Persistent workspace terminal closed."

        return [
            StructuredTool.from_function(
                func=run_command,
                name="terminal_run",
                description=(
                    "Run an approved shell command in the persistent workspace terminal. "
                    "The shell retains its working directory and environment between calls. "
                    "If a command exceeds its timeout, call terminal_read to collect its result."
                ),
                args_schema=TerminalCommandArgs,
            ),
            StructuredTool.from_function(
                func=read_output,
                name="terminal_read",
                description=(
                    "Read output from a still-running command in the persistent workspace terminal."
                ),
                args_schema=TerminalReadArgs,
            ),
            StructuredTool.from_function(
                func=close_terminal,
                name="terminal_close",
                description="Close the persistent workspace terminal process.",
            ),
        ]

    def _shell_command(self) -> list[str]:
        if os.name == "nt":
            executable = shutil.which("pwsh") or shutil.which("powershell.exe")
            if executable is None:
                raise FileNotFoundError("PowerShell (pwsh or powershell.exe) was not found.")
            bootstrap = (
                "$ErrorActionPreference = 'Continue'\n"
                "while ($null -ne ($__aica_line = [Console]::In.ReadLine())) {\n"
                "  $__aica_request = ConvertFrom-Json $__aica_line\n"
                "  $__aica_script = [Text.Encoding]::UTF8.GetString("
                "[Convert]::FromBase64String($__aica_request.command))\n"
                "  $__aica_old_error = $ErrorActionPreference\n"
                "  $ErrorActionPreference = 'Stop'\n"
                "  $global:LASTEXITCODE = 0\n"
                "  try {\n"
                "    $__aica_output = Invoke-Expression $__aica_script *>&1 | Out-String\n"
                "    $__aica_exit_code = if ($LASTEXITCODE -ne 0) { $LASTEXITCODE } else { 0 }\n"
                "  } catch {\n"
                "    $__aica_output = ($_ | Out-String)\n"
                "    $__aica_exit_code = 1\n"
                "  } finally {\n"
                "    $ErrorActionPreference = $__aica_old_error\n"
                "  }\n"
                "  [Console]::Out.WriteLine($__aica_output)\n"
                "  [Console]::Out.WriteLine((($__aica_request.marker) + ':' + $__aica_exit_code))\n"
                "}\n"
            )
            return [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", bootstrap]

        executable = shutil.which("bash")
        if executable is None:
            raise FileNotFoundError("bash was not found.")
        return [executable, "--noprofile", "--norc"]

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self._output_queue = queue.Queue()
        self._process = subprocess.Popen(
            self._shell_command(),
            cwd=self._root_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self._process.stdout is None:
            raise RuntimeError("Terminal process stdout was not created.")

        self._reader = threading.Thread(
            target=self._read_process_output,
            args=(self._process.stdout,),
            name=f"terminal-output-{self._session_id[:8]}",
            daemon=True,
        )
        self._reader.start()

    def _read_process_output(self, stream: Any) -> None:
        for line in stream:
            self._output_queue.put(line)
        self._output_queue.put(None)

    def _wrap_command(self, command: str, marker: str) -> str:
        if self._platform == "powershell":
            encoded_command = base64.b64encode(command.encode("utf-8")).decode("ascii")
            return json.dumps({"command": encoded_command, "marker": marker}) + "\n"
        return (
            f"{command}\n"
            "__aica_exit_code=$?\n"
            f"printf '\\n{marker}:%s\\n' \"$__aica_exit_code\"\n"
        )

    def _collect(self, marker: str, timeout_seconds: int) -> tuple[bool, int | None, str]:
        deadline = time.monotonic() + timeout_seconds
        output = self._pending_output
        if self._pending_marker != marker:
            output = []

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                line = self._output_queue.get(timeout=remaining)
                if line is None:
                    process_code = self._process.poll() if self._process else None
                    raise RuntimeError(
                        f"Terminal process exited before command completion (exit code {process_code}). "
                        + "".join(output)
                    )

                match = re.fullmatch(re.escape(marker) + r":(-?\d+)\s*", line.strip())
                if match:
                    self._pending_marker = None
                    self._pending_output = []
                    return True, int(match.group(1)), "".join(output).strip()
                output.append(line)
        except queue.Empty:
            self._pending_marker = marker
            self._pending_output = output
            return False, None, "".join(output).strip()

    @staticmethod
    def _format_result(completed: bool, exit_code: int | None, output: str, session_id: str) -> str:
        if not completed:
            detail = output or "(no output yet)"
            return (
                f"[Terminal command still running; session={session_id}]\n"
                f"{detail}\nCall terminal_read to wait for completion."
            )
        status = "succeeded" if exit_code == 0 else f"failed with exit code {exit_code}"
        prefix = f"[Command {status}; terminal session={session_id}]"
        return f"{prefix}\n{output}" if output else prefix

    def run(self, command: str, *, timeout_seconds: int = 30) -> str:
        command = command.strip()
        if not command:
            raise ValueError("Terminal command cannot be empty.")
        if not 1 <= timeout_seconds <= 1800:
            raise ValueError("timeout_seconds must be between 1 and 1800.")

        with self._lock:
            if self._pending_marker is not None:
                raise RuntimeError(
                    "A terminal command is still running. Call terminal_read before starting another command."
                )
            try:
                self._guard.enforce(
                    f"Bash({command})",
                    interrupt_fn=interrupt,
                    description=f"Persistent terminal requests: {command}",
                )
            except PermissionError as exc:
                return str(exc)

            self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise RuntimeError("Persistent terminal stdin is unavailable.")

            marker = f"__AICA_DONE_{uuid.uuid4().hex}"
            self._pending_marker = marker
            self._pending_output = []
            try:
                process.stdin.write(self._wrap_command(command, marker))
                process.stdin.flush()
            except OSError as exc:
                self._pending_marker = None
                raise RuntimeError(f"Could not send command to terminal: {exc}") from exc
            completed, exit_code, output = self._collect(marker, timeout_seconds)
            return self._format_result(completed, exit_code, output, self._session_id)

    def read(self, *, wait_seconds: int = 2) -> str:
        if not 0 <= wait_seconds <= 30:
            raise ValueError("wait_seconds must be between 0 and 30.")

        with self._lock:
            marker = self._pending_marker
            if marker is None:
                return f"No command is currently running (terminal session={self._session_id})."
            completed, exit_code, output = self._collect(marker, wait_seconds)
            return self._format_result(completed, exit_code, output, self._session_id)

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            self._pending_marker = None
            self._pending_output = []
            if process is None:
                return

            if process.poll() is None:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
