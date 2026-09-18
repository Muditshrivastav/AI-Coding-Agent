"""
tests/test_harness_guard.py - Comprehensive unit tests for HarnessGuard (agent/guardrail.py).

Covers:
  - check_permission: allow / ask / deny / default fallback
  - enforce: Gate 1 (deny), Gate 2 (ask + rejected / approved), Gate 3 (allow)
  - log_failure: structured row appended to failures.md
  - load / _reload_if_changed: hot-reload on file change
  - wrap_tool_with_guard: guarded StructuredTool wrapper
  - whitespace normalisation (bypass-prevention)
  - missing / malformed permissions.json edge cases
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.guardrail import HarnessGuard, _append_failure


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_permissions(tmp_path: Path, rules: dict) -> Path:
    """Write a permissions.json to tmp_path/harness/ and return its path."""
    harness = tmp_path / "harness"
    harness.mkdir(parents=True, exist_ok=True)
    perm_file = harness / "permissions.json"
    perm_file.write_text(json.dumps(rules), encoding="utf-8")
    return perm_file


def _make_guard(tmp_path: Path, rules: dict | None = None) -> tuple[HarnessGuard, Path, Path]:
    """Return (guard, permissions_path, failures_path)."""
    if rules is None:
        rules = {
            "allow": ["Bash(pytest*)", "Read(src/**)"],
            "ask": ["Bash(git push*)", "vercel_deploy"],
            "deny": ["Read(.env)", "Bash(*rm -rf*)"],
        }
    perm_file = _make_permissions(tmp_path, rules)
    failures_file = tmp_path / "harness" / "failures.md"
    guard = HarnessGuard(
        permissions_path=str(perm_file),
        failures_path=str(failures_file),
    )
    return guard, perm_file, failures_file


# ===========================================================================
# 1. check_permission - allow / ask / deny / default fallback
# ===========================================================================


class TestCheckPermission:
    def test_allow_pattern_matched(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(pytest -q)") == "allow"

    def test_allow_wildcard_read(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Read(src/app.py)") == "allow"

    def test_deny_dotenv(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Read(.env)") == "deny"

    def test_deny_rm_rf(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(sudo rm -rf /)") == "deny"

    def test_ask_git_push(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(git push origin main)") == "ask"

    def test_ask_vercel(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("vercel_deploy") == "ask"

    def test_default_unknown_is_ask(self, tmp_path):
        """Unrecognised action must default to 'ask' (fail-safe)."""
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("some_random_unknown_action") == "ask"

    def test_deny_has_priority_over_allow(self, tmp_path):
        """A signature matching both deny and allow must resolve to deny."""
        rules = {
            "allow": ["Bash(*rm*)"],
            "ask": [],
            "deny": ["Bash(*rm -rf*)"],
        }
        guard, _, _ = _make_guard(tmp_path, rules)
        assert guard.check_permission("Bash(rm -rf /tmp)") == "deny"

    def test_whitespace_normalisation_bypasses_deny(self, tmp_path):
        """Extra padding around 'rm -rf' must still be caught by the deny rule."""
        guard, _, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(   rm  -rf  /)") == "deny"

    def test_uses_actual_permissions_json(self):
        """Smoke-test against the real harness/permissions.json if it exists."""
        real_perm = Path("harness/permissions.json")
        real_fail = Path("harness/failures.md")
        if not real_perm.exists():
            pytest.skip("harness/permissions.json not present in working directory")
        guard = HarnessGuard(str(real_perm), failures_path=str(real_fail))
        assert guard.check_permission("Bash(ruff check .)") == "allow"
        assert guard.check_permission("Read(.env)") == "deny"
        assert guard.check_permission("Bash(git push origin main)") == "ask"


# ===========================================================================
# 2. load / _reload_if_changed
# ===========================================================================


class TestLoadAndHotReload:
    def test_load_missing_file_does_not_raise(self, tmp_path):
        """HarnessGuard must not crash if permissions.json does not exist."""
        guard = HarnessGuard(
            permissions_path=str(tmp_path / "nonexistent.json"),
            failures_path=str(tmp_path / "failures.md"),
        )
        assert guard.check_permission("anything") == "ask"

    def test_load_malformed_json_keeps_previous_rules(self, tmp_path):
        """A mid-write corrupt file must not wipe previously loaded rules."""
        guard, perm_file, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(pytest -q)") == "allow"
        perm_file.write_text("{invalid json", encoding="utf-8")
        guard.load()
        assert guard.check_permission("Bash(pytest -q)") == "allow"

    def test_load_wrong_schema_top_level_list(self, tmp_path):
        """A JSON array at the top level is rejected; previous rules kept."""
        guard, perm_file, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(pytest -q)") == "allow"
        perm_file.write_text(json.dumps([{"allow": ["*"]}]), encoding="utf-8")
        guard.load()
        assert guard.check_permission("Bash(pytest -q)") == "allow"

    def test_load_wrong_schema_non_list_value(self, tmp_path):
        """A non-list value for a key is rejected; previous rules kept."""
        guard, perm_file, _ = _make_guard(tmp_path)
        perm_file.write_text(
            json.dumps({"allow": "Bash(*)", "ask": [], "deny": []}),
            encoding="utf-8",
        )
        guard.load()
        assert guard.check_permission("Bash(pytest -q)") == "allow"

    def test_hot_reload_picks_up_new_rules(self, tmp_path):
        """_reload_if_changed must pick up a permissions.json update."""
        guard, perm_file, _ = _make_guard(tmp_path)
        assert guard.check_permission("Bash(pytest -q)") == "allow"

        time.sleep(0.01)  # ensure mtime differs
        new_rules = {"allow": [], "ask": [], "deny": ["Bash(pytest*)"]}
        perm_file.write_text(json.dumps(new_rules), encoding="utf-8")
        perm_file.touch()

        guard._reload_if_changed()
        assert guard.check_permission("Bash(pytest -q)") == "deny"

    def test_reload_not_triggered_when_mtime_unchanged(self, tmp_path):
        """Rules must NOT change if the file has not been modified."""
        guard, perm_file, _ = _make_guard(tmp_path)
        original_permissions = dict(guard._permissions)
        guard._reload_if_changed()
        assert guard._permissions == original_permissions


# ===========================================================================
# 3. enforce - Gates 1, 2, 3
# ===========================================================================


class TestEnforce:
    # -- Gate 1: deny --------------------------------------------------------

    def test_enforce_deny_raises_permission_error(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        with pytest.raises(PermissionError, match="SECURITY DENIED"):
            guard.enforce("Read(.env)")

    def test_enforce_deny_logs_to_failures_md(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        with pytest.raises(PermissionError):
            guard.enforce("Read(.env)")
        content = failures_file.read_text(encoding="utf-8")
        assert "security-deny" in content
        assert "Read(.env)" in content

    # -- Gate 2: ask + rejected ----------------------------------------------

    def test_enforce_ask_with_bool_false_rejects(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        interrupt_fn = MagicMock(return_value=False)
        with pytest.raises(PermissionError, match="HITL REJECTED"):
            guard.enforce("Bash(git push origin main)", interrupt_fn=interrupt_fn)
        interrupt_fn.assert_called_once()

    def test_enforce_ask_dict_rejected_logs(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        interrupt_fn = MagicMock(return_value={"approved": False})
        with pytest.raises(PermissionError):
            guard.enforce("vercel_deploy", interrupt_fn=interrupt_fn)
        content = failures_file.read_text(encoding="utf-8")
        assert "human-rejected" in content

    def test_enforce_ask_string_no_rejected(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        interrupt_fn = MagicMock(return_value="no")
        with pytest.raises(PermissionError, match="HITL REJECTED"):
            guard.enforce("vercel_deploy", interrupt_fn=interrupt_fn)

    # -- Gate 2: ask + approved ----------------------------------------------

    def test_enforce_ask_bool_true_approved(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        interrupt_fn = MagicMock(return_value=True)
        guard.enforce("vercel_deploy", interrupt_fn=interrupt_fn)  # must not raise
        interrupt_fn.assert_called_once()

    def test_enforce_ask_dict_approved(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        interrupt_fn = MagicMock(return_value={"approved": True})
        guard.enforce("Bash(git push origin main)", interrupt_fn=interrupt_fn)

    def test_enforce_ask_string_approved(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        for resp in ("yes", "approve", "approved", "True"):
            interrupt_fn = MagicMock(return_value=resp)
            guard.enforce("vercel_deploy", interrupt_fn=interrupt_fn)

    # -- Gate 3: allow -------------------------------------------------------

    def test_enforce_allow_passes_without_interrupt(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        interrupt_fn = MagicMock()
        guard.enforce("Bash(pytest -q)", interrupt_fn=interrupt_fn)
        interrupt_fn.assert_not_called()

    def test_enforce_description_overrides_label_in_log(self, tmp_path):
        """description kwarg should appear in failures.md row instead of raw signature."""
        guard, _, failures_file = _make_guard(tmp_path)
        with pytest.raises(PermissionError):
            guard.enforce("Read(.env)", description="Trying to read secrets")
        content = failures_file.read_text(encoding="utf-8")
        assert "Trying to read secrets" in content

    def test_enforce_extra_context_passed_to_interrupt(self, tmp_path):
        """extra_context dict should be merged into the interrupt payload."""
        guard, _, _ = _make_guard(tmp_path)
        captured: dict = {}

        def fake_interrupt(payload):
            captured.update(payload)
            return True

        guard.enforce(
            "vercel_deploy",
            interrupt_fn=fake_interrupt,
            extra_context={"env": "staging"},
        )
        assert captured.get("env") == "staging"
        assert captured.get("signature") == "vercel_deploy"


# ===========================================================================
# 4. log_failure
# ===========================================================================


class TestLogFailure:
    def test_log_failure_creates_file_with_header(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        guard.log_failure(
            stage="verification",
            category="test-failure",
            description="pytest returned exit code 1",
            resolution="Agent retrying fix.",
        )
        content = failures_file.read_text(encoding="utf-8")
        assert "# Failure Log" in content
        assert "verification" in content
        assert "test-failure" in content
        assert "pytest returned exit code 1" in content
        assert "Agent retrying fix." in content

    def test_log_failure_appends_multiple_rows(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        for i in range(3):
            guard.log_failure("stage", "cat", f"error {i}")
        rows = [
            line
            for line in failures_file.read_text(encoding="utf-8").splitlines()
            if line.startswith("|") and "---" not in line and "Timestamp" not in line
        ]
        assert len(rows) == 3

    def test_log_failure_reinitialises_empty_file(self, tmp_path):
        guard, _, failures_file = _make_guard(tmp_path)
        failures_file.write_text("", encoding="utf-8")
        guard.log_failure("stage", "cat", "desc")
        content = failures_file.read_text(encoding="utf-8")
        assert "# Failure Log" in content

    def test_append_failure_helper_default_resolution(self, tmp_path):
        """_append_failure standalone helper should use em-dash as default resolution."""
        failures_file = tmp_path / "failures.md"
        _append_failure(failures_file, "s", "c", "description")
        content = failures_file.read_text(encoding="utf-8")
        assert "\u2014" in content


# ===========================================================================
# 5. wrap_tool_with_guard
# ===========================================================================


class TestWrapToolWithGuard:
    def _make_tool(self, name: str, description: str = "test tool"):
        from langchain_core.tools import StructuredTool

        def echo(text: str = "") -> str:
            return f"echo:{text}"

        return StructuredTool.from_function(func=echo, name=name, description=description)

    def test_wrapped_tool_passes_on_allow(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        tool = self._make_tool("Bash(pytest -q)")
        wrapped = guard.wrap_tool_with_guard(tool, interrupt_fn=None)
        result = wrapped.func(text="hello")
        assert result == "echo:hello"

    def test_wrapped_tool_blocks_on_deny(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        from langchain_core.tools import StructuredTool

        def noop():
            return "ok"

        tool = StructuredTool.from_function(func=noop, name="Read(.env)", description="bad")
        wrapped = guard.wrap_tool_with_guard(tool, interrupt_fn=None)
        with pytest.raises(PermissionError):
            wrapped.func()

    def test_wrapped_tool_hitl_approved_executes(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        from langchain_core.tools import StructuredTool

        def deploy():
            return "deployed"

        tool = StructuredTool.from_function(func=deploy, name="vercel_deploy", description="deploy")
        interrupt_fn = MagicMock(return_value=True)
        wrapped = guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt_fn)
        result = wrapped.func()
        assert result == "deployed"
        interrupt_fn.assert_called_once()

    def test_wrap_tool_without_func_returns_original(self, tmp_path):
        """A tool without a .func attribute must be returned unchanged."""
        guard, _, _ = _make_guard(tmp_path)
        fake_tool = MagicMock(spec=[])  # no .func attribute
        result = guard.wrap_tool_with_guard(fake_tool)
        assert result is fake_tool

    def test_wrapped_tool_description_has_hitl_prefix(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        tool = self._make_tool("Bash(pytest -q)", description="Run tests")
        wrapped = guard.wrap_tool_with_guard(tool)
        assert wrapped.description.startswith("[HITL-guarded]")

    def test_wrapped_tool_preserves_name(self, tmp_path):
        guard, _, _ = _make_guard(tmp_path)
        tool = self._make_tool("Bash(pytest -q)")
        wrapped = guard.wrap_tool_with_guard(tool)
        assert wrapped.name == tool.name
