"""
tests/test_docker_sandbox.py - Comprehensive unit tests for DockerSandboxBackend.

Covers:
  - __init__: default params, custom params, abs_root resolution
  - id property: correct format
  - is_docker_available: caches result, handles missing binary, handles daemon down,
                         handles daemon up, handles exception from subprocess
  - build_docker_command: volume mount format, security flags, image and command placement,
                          Windows path conversion, custom workspace/image/limits
  - execute: Docker available path (success, non-zero exit, timeout, generic exception),
             fallback path (Docker unavailable + auto_fallback=True runs LocalShellBackend),
             no-fallback path (Docker unavailable + auto_fallback=False returns error response),
             timeout parameter forwarded correctly
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents.backends.protocol import ExecuteResponse
from frameworks.docker_sandbox import DockerSandboxBackend

# Root for all temporary directories created by this test module.
# Using a workspace-local path avoids the sandbox PermissionError on %LOCALAPPDATA%\Temp.
_TMP_ROOT = Path(__file__).parent / "_tmp"


@pytest.fixture()
def workspace_tmp():
    """Provide a fresh, isolated temp directory inside the workspace tree."""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=_TMP_ROOT))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(tmp_path: Path, **kwargs) -> DockerSandboxBackend:
    """Helper: instantiate DockerSandboxBackend pointing at tmp_path."""
    defaults = dict(
        root_dir=str(workspace_tmp),
        image="python:3.12-slim",
        container_workspace="/workspace",
        memory_limit="1g",
        cpu_limit="1.0",
        cap_drop="ALL",
        network="bridge",
        timeout=30,
        auto_fallback=True,
    )
    defaults.update(kwargs)
    return DockerSandboxBackend(**defaults)


# ===========================================================================
# 1. __init__ and id property
# ===========================================================================


class TestInit:
    def test_defaults_set_correctly(self, workspace_tmp):
        b = DockerSandboxBackend(root_dir=str(workspace_tmp))
        assert b._image in ("ai-coding-agent-sandbox:latest", "python:3.12-slim")
        assert b._container_workspace == "/workspace"
        assert b._memory_limit == "1g"
        assert b._cpu_limit == "1.0"
        assert b._cap_drop in (["ALL"], "ALL")
        assert b._network == "bridge"
        assert b._auto_fallback is True
        assert b._docker_available is None

    def test_custom_params_stored(self, workspace_tmp):
        b = DockerSandboxBackend(
            root_dir=str(workspace_tmp),
            image="ubuntu:22.04",
            container_workspace="/project",
            memory_limit="512m",
            cpu_limit="0.5",
            cap_drop="NET_ADMIN",
            network="none",
            timeout=60,
            auto_fallback=False,
        )
        assert b._image == "ubuntu:22.04"
        assert b._container_workspace == "/project"
        assert b._memory_limit == "512m"
        assert b._cpu_limit == "0.5"
        assert b._cap_drop in (["NET_ADMIN"], "NET_ADMIN")
        assert b._network == "none"
        assert b._auto_fallback is False

    def test_abs_root_is_resolved(self, workspace_tmp):
        """_abs_root must be an absolute, resolved path regardless of input form."""
        b = DockerSandboxBackend(root_dir=str(workspace_tmp))
        assert Path(b._abs_root).is_absolute()
        assert b._abs_root == str(workspace_tmp.resolve())

    def test_id_property_format(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        assert b.id.startswith("docker-sandbox:")
        assert b._abs_root in b.id


# ===========================================================================
# 2. is_docker_available
# ===========================================================================


class TestIsDockerAvailable:
    def test_returns_cached_true(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        b._docker_available = True
        # Should not touch shutil.which or subprocess at all
        with patch("shutil.which") as mock_which:
            result = b.is_docker_available
        assert result is True
        mock_which.assert_not_called()

    def test_returns_cached_false(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        b._docker_available = False
        with patch("shutil.which") as mock_which:
            result = b.is_docker_available
        assert result is False
        mock_which.assert_not_called()

    def test_missing_docker_binary_returns_false(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        with patch("frameworks.docker_sandbox.shutil.which", return_value=None):
            result = b.is_docker_available
        assert result is False
        assert b._docker_available is False

    def test_daemon_up_returns_true(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with (
            patch("frameworks.docker_sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc),
        ):
            result = b.is_docker_available
        assert result is True
        assert b._docker_available is True

    def test_daemon_down_returns_false(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with (
            patch("frameworks.docker_sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc),
        ):
            result = b.is_docker_available
        assert result is False
        assert b._docker_available is False

    def test_subprocess_exception_returns_false(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        with (
            patch("frameworks.docker_sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch(
                "frameworks.docker_sandbox.subprocess.run",
                side_effect=FileNotFoundError("docker not found"),
            ),
        ):
            result = b.is_docker_available
        assert result is False
        assert b._docker_available is False

    def test_result_is_cached_after_first_probe(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        mock_proc = MagicMock(returncode=0)
        with (
            patch("frameworks.docker_sandbox.shutil.which", return_value="/usr/bin/docker"),
            patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run,
        ):
            _ = b.is_docker_available
            _ = b.is_docker_available  # second call
        assert mock_run.call_count == 1  # subprocess.run called only once


# ===========================================================================
# 3. build_docker_command
# ===========================================================================


class TestBuildDockerCommand:
    def test_starts_with_docker_run_rm(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        cmd = b.build_docker_command("echo hello")
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd

    def test_volume_mount_contains_abs_root_and_workspace(self, workspace_tmp):
        b = _make_backend(workspace_tmp, container_workspace="/workspace")
        cmd = b.build_docker_command("ls")
        # -v flag followed by <src>:/workspace:rw
        v_index = cmd.index("-v")
        mount_str = cmd[v_index + 1]
        # workspace part
        assert ":/workspace:rw" in mount_str
        # host part (forward-slash form for Docker compat on Windows)
        abs_root_fwd = b._abs_root.replace("\\", "/")
        assert abs_root_fwd in mount_str

    def test_working_dir_flag(self, workspace_tmp):
        b = _make_backend(workspace_tmp, container_workspace="/project")
        cmd = b.build_docker_command("pwd")
        w_index = cmd.index("-w")
        assert cmd[w_index + 1] == "/project"

    def test_memory_limit_flag(self, workspace_tmp):
        b = _make_backend(workspace_tmp, memory_limit="512m")
        cmd = b.build_docker_command("ls")
        assert "--memory=512m" in cmd

    def test_cpu_limit_flag(self, workspace_tmp):
        b = _make_backend(workspace_tmp, cpu_limit="0.5")
        cmd = b.build_docker_command("ls")
        assert "--cpus=0.5" in cmd

    def test_cap_drop_flag(self, workspace_tmp):
        b = _make_backend(workspace_tmp, cap_drop="ALL")
        cmd = b.build_docker_command("ls")
        assert "--cap-drop=ALL" in cmd

    def test_network_flag(self, workspace_tmp):
        b = _make_backend(workspace_tmp, network="none")
        cmd = b.build_docker_command("ls")
        assert "--network=none" in cmd

    def test_image_placed_before_sh_c(self, workspace_tmp):
        b = _make_backend(workspace_tmp, image="ubuntu:22.04")
        cmd = b.build_docker_command("echo test")
        img_index = cmd.index("ubuntu:22.04")
        sh_index = cmd.index("sh")
        assert img_index < sh_index

    def test_command_is_last_element(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        cmd = b.build_docker_command("echo hello world")
        assert cmd[-1] == "echo hello world"

    def test_sh_c_precedes_command(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        cmd = b.build_docker_command("ls -la")
        sh_index = cmd.index("sh")
        assert cmd[sh_index + 1] == "-c"
        assert cmd[sh_index + 2] == "ls -la"

    def test_windows_backslash_path_converted(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        # Simulate a Windows abs_root
        b._abs_root = "D:\\My Projects\\code"
        cmd = b.build_docker_command("ls")
        v_index = cmd.index("-v")
        mount_str = cmd[v_index + 1]
        assert "\\" not in mount_str
        assert "D:/My Projects/code" in mount_str

    def test_custom_workspace_reflected_in_mount_and_workdir(self, workspace_tmp):
        b = _make_backend(workspace_tmp, container_workspace="/custom")
        cmd = b.build_docker_command("pwd")
        v_index = cmd.index("-v")
        mount_str = cmd[v_index + 1]
        w_index = cmd.index("-w")
        assert ":/custom:rw" in mount_str
        assert cmd[w_index + 1] == "/custom"


# ===========================================================================
# 4. execute — Docker available path
# ===========================================================================


class TestExecuteDockerAvailable:
    def _patch_docker(self, workspace_tmp, stdout: str, returncode: int = 0):
        """Return a backend and patched context where Docker is available and returns given output."""
        b = _make_backend(workspace_tmp, auto_fallback=True)
        b._docker_available = True

        mock_proc = MagicMock()
        mock_proc.stdout = stdout
        mock_proc.returncode = returncode
        return b, mock_proc

    def test_success_returns_output_and_exit_0(self, workspace_tmp):
        b, mock_proc = self._patch_docker(workspace_tmp, stdout="hello\n", returncode=0)
        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc):
            res = b.execute("echo hello")
        assert isinstance(res, ExecuteResponse)
        assert res.exit_code == 0
        assert "hello" in res.output

    def test_nonzero_exit_code_preserved(self, workspace_tmp):
        b, mock_proc = self._patch_docker(workspace_tmp, stdout="error output", returncode=2)
        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc):
            res = b.execute("false")
        assert res.exit_code == 2
        assert "error output" in res.output

    def test_command_forwarded_to_docker_invocation(self, workspace_tmp):
        b, mock_proc = self._patch_docker(workspace_tmp, stdout="", returncode=0)
        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run:
            b.execute("echo from test")
        called_cmd = mock_run.call_args[0][0]
        assert "echo from test" in called_cmd

    def test_timeout_expired_returns_exit_124(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        b._docker_available = True
        exc = subprocess.TimeoutExpired(cmd=["docker"], timeout=30)
        exc.stdout = "partial output"
        with patch("frameworks.docker_sandbox.subprocess.run", side_effect=exc):
            res = b.execute("sleep 999")
        assert res.exit_code == 124
        assert "timed out" in res.output.lower()

    def test_timeout_expired_without_stdout_handled(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        b._docker_available = True
        exc = subprocess.TimeoutExpired(cmd=["docker"], timeout=30)
        exc.stdout = None
        with patch("frameworks.docker_sandbox.subprocess.run", side_effect=exc):
            res = b.execute("sleep 999")
        assert res.exit_code == 124

    def test_generic_exception_returns_exit_1_with_message(self, workspace_tmp):
        b = _make_backend(workspace_tmp)
        b._docker_available = True
        with patch(
            "frameworks.docker_sandbox.subprocess.run",
            side_effect=OSError("docker not found on PATH"),
        ):
            res = b.execute("ls")
        assert res.exit_code == 1
        assert "Error executing command in Docker sandbox" in res.output

    def test_custom_timeout_forwarded(self, workspace_tmp):
        b = _make_backend(workspace_tmp, timeout=60)
        b._docker_available = True
        mock_proc = MagicMock(stdout="ok", returncode=0)
        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run:
            b.execute("ls", timeout=5)
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 5

    def test_default_instance_timeout_used_when_none(self, workspace_tmp):
        b = _make_backend(workspace_tmp, timeout=45)
        b._docker_available = True
        mock_proc = MagicMock(stdout="ok", returncode=0)
        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run:
            b.execute("ls")  # no explicit timeout
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 45


# ===========================================================================
# 5. execute — fallback path (Docker unavailable)
# ===========================================================================


class TestExecuteFallback:
    def test_fallback_enabled_calls_local_shell(self, workspace_tmp):
        """When Docker is down and auto_fallback=True, the local shell backend runs instead."""
        b = _make_backend(workspace_tmp, auto_fallback=True)
        b._docker_available = False

        fake_response = ExecuteResponse(output="fallback output", exit_code=0)
        with patch.object(
            b.__class__.__bases__[0],  # LocalShellBackend
            "execute",
            return_value=fake_response,
        ):
            res = b.execute("echo from fallback")
        assert res.output == "fallback output"
        assert res.exit_code == 0

    def test_no_fallback_returns_error_response(self, workspace_tmp):
        """When Docker is down and auto_fallback=False, no subprocess is run — returns error."""
        b = _make_backend(workspace_tmp, auto_fallback=False)
        b._docker_available = False

        with patch("frameworks.docker_sandbox.subprocess.run") as mock_run:
            res = b.execute("ls")
        mock_run.assert_not_called()
        assert res.exit_code == 1
        assert "Docker daemon" in res.output or "not running" in res.output.lower()

    def test_fallback_does_not_call_docker(self, workspace_tmp):
        """With fallback active, docker CLI is never invoked."""
        b = _make_backend(workspace_tmp, auto_fallback=True)
        b._docker_available = False

        fake_response = ExecuteResponse(output="local", exit_code=0)
        with (
            patch.object(b.__class__.__bases__[0], "execute", return_value=fake_response),
            patch("frameworks.docker_sandbox.subprocess.run") as mock_run,
        ):
            b.execute("echo test")
        mock_run.assert_not_called()

    def test_docker_available_skips_fallback(self, workspace_tmp):
        """If Docker IS available, the local shell fallback path must never execute."""
        b = _make_backend(workspace_tmp, auto_fallback=True)
        b._docker_available = True
        mock_proc = MagicMock(stdout="docker output", returncode=0)

        with (
            patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc),
            patch.object(
                b.__class__.__bases__[0],
                "execute",
                side_effect=AssertionError("should not call local shell"),
            ),
        ):
            res = b.execute("ls")
        assert "docker output" in res.output


# ===========================================================================
# 6. Integration-style: build_docker_command consistency with execute
# ===========================================================================


class TestBuildCommandConsistency:
    def test_command_built_by_helper_is_what_is_run(self, workspace_tmp):
        """build_docker_command must produce exactly the invocation passed to subprocess.run."""
        b = _make_backend(workspace_tmp)
        b._docker_available = True
        mock_proc = MagicMock(stdout="ok", returncode=0)

        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run:
            b.execute("pytest -q")

        expected_cmd = b.build_docker_command("pytest -q")
        actual_cmd = mock_run.call_args[0][0]
        assert actual_cmd == expected_cmd

    def test_all_security_flags_present_in_execute(self, workspace_tmp):
        """Security flags set at construction appear in the actual docker invocation."""
        b = DockerSandboxBackend(
            root_dir=str(workspace_tmp),
            memory_limit="256m",
            cpu_limit="0.25",
            cap_drop="ALL",
            network="none",
        )
        b._docker_available = True
        mock_proc = MagicMock(stdout="ok", returncode=0)

        with patch("frameworks.docker_sandbox.subprocess.run", return_value=mock_proc) as mock_run:
            b.execute("ls")

        cmd = mock_run.call_args[0][0]
        assert "--memory=256m" in cmd
        assert "--cpus=0.25" in cmd
        assert "--cap-drop=ALL" in cmd
        assert "--network=none" in cmd
