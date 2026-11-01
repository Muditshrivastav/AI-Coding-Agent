import json
import shlex
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.core import CodingAgentHarness
from agent.guardrail import HarnessGuard
from frameworks.terminal_session import TerminalSessionManager


def _guard(tmp_path, *, allow=None, ask=None):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir(exist_ok=True)
    permissions_path = harness_dir / "permissions.json"
    permissions_path.write_text(
        json.dumps({"allow": allow or [], "ask": ask or [], "deny": []}),
        encoding="utf-8",
    )
    return HarnessGuard(str(permissions_path), silent_mode=True)


def _quote_shell_path(path, platform):
    if platform == "powershell":
        return "'" + str(path).replace("'", "''") + "'"
    return shlex.quote(str(path))


def test_terminal_session_preserves_working_directory(tmp_path):
    guard = _guard(tmp_path, allow=["Bash(*)"])
    manager = TerminalSessionManager(str(tmp_path), guard)
    try:
        target_dir = tmp_path / "nested"
        target_dir.mkdir()
        if manager._platform == "powershell":
            change_dir = f"Set-Location -LiteralPath {_quote_shell_path(target_dir, manager._platform)}"
        else:
            change_dir = f"cd {_quote_shell_path(target_dir, manager._platform)}"

        assert "succeeded" in manager.run(change_dir)
        command = "Get-Location | ForEach-Object {$_.Path}" if manager._platform == "powershell" else "pwd"
        result = manager.run(command)

        assert "succeeded" in result
        assert str(target_dir) in result
    finally:
        manager.close()


def test_terminal_session_requests_approval_before_starting_process(tmp_path, monkeypatch):
    guard = _guard(tmp_path, ask=["Bash(*)"])
    manager = TerminalSessionManager(str(tmp_path), guard)
    approval_requests = []

    def approve(request):
        approval_requests.append(request)
        return {"approved": True}

    monkeypatch.setattr("frameworks.terminal_session.interrupt", approve)
    try:
        result = manager.run("echo approved")
        assert "succeeded" in result
        assert approval_requests[0]["signature"] == "Bash(echo approved)"
        assert manager._process is not None
    finally:
        manager.close()


def test_terminal_session_denial_does_not_start_process(tmp_path, monkeypatch):
    guard = _guard(tmp_path, ask=["Bash(*)"])
    manager = TerminalSessionManager(str(tmp_path), guard)
    approval_requests = []

    def deny(request):
        approval_requests.append(request)
        return {"approved": False}

    monkeypatch.setattr("frameworks.terminal_session.interrupt", deny)

    result = manager.run("echo should-not-run")

    assert "HITL REJECTED" in result
    assert approval_requests[0]["signature"] == "Bash(echo should-not-run)"
    assert manager._process is None


def test_terminal_read_collects_timed_out_command(tmp_path):
    guard = _guard(tmp_path, allow=["Bash(*)"])
    manager = TerminalSessionManager(str(tmp_path), guard)
    try:
        command = (
            "Start-Sleep -Seconds 2; Write-Output complete"
            if manager._platform == "powershell"
            else "sleep 2; echo complete"
        )
        result = manager.run(command, timeout_seconds=1)

        assert "still running" in result
        completed = manager.read(wait_seconds=20)

        assert "succeeded" in completed
        assert "complete" in completed
    finally:
        manager.close()


def test_root_model_build_exposes_guarded_terminal_tools(tmp_path, monkeypatch):
    guard = _guard(
        tmp_path,
        allow=["terminal_run", "terminal_read", "terminal_close"],
        ask=["Bash(*)"],
    )
    terminal_tools = TerminalSessionManager(str(tmp_path), guard).get_tools()
    captured = {}
    approval_requests = []

    def approve(request):
        approval_requests.append(request)
        return {"approved": True}

    monkeypatch.setattr("frameworks.terminal_session.interrupt", approve)

    harness = CodingAgentHarness.__new__(CodingAgentHarness)
    setattr(harness, "_model", "ollama:default")
    setattr(harness, "_tools", terminal_tools)
    setattr(harness, "_guard", guard)
    setattr(harness, "_external_tools", object())
    setattr(harness, "_root_dir", str(tmp_path))
    setattr(harness, "_system_prompt", "")
    setattr(harness, "_backend", SimpleNamespace(
        backend=object(),
        build_agent=lambda **kwargs: captured.update(kwargs) or kwargs,
    ))
    setattr(harness, "_runtime", SimpleNamespace(checkpointer=None))

    monkeypatch.setattr("agent.core.build_dev_subagent", MagicMock(return_value={}))
    harness.build_agent_for_model("ollama:alternate")

    root_tool_names = {tool.name for tool in captured["tools"]}
    assert {"terminal_run", "terminal_read", "terminal_close"} <= root_tool_names
    assert all(tool.description.startswith("[HITL-guarded]") for tool in captured["tools"])
    run_tool = next(tool for tool in captured["tools"] if tool.name == "terminal_run")
    close_tool = next(tool for tool in captured["tools"] if tool.name == "terminal_close")
    try:
        result = run_tool.invoke({"command": "echo root-tool"})

        assert "succeeded" in result
        assert approval_requests[0]["signature"] == "Bash(echo root-tool)"
    finally:
        close_tool.invoke({})
