"""
tests/test_verification_tools.py - Unit tests for VerificationToolset and run_verification tool.
"""

import pytest
from unittest.mock import patch, MagicMock
from tools.verification_tools import VerificationToolset, create_verification_tool


def test_verification_tool_initialization():
    tool = create_verification_tool()
    assert tool.name == "run_verification"
    assert "make verify" in tool.description.lower() or "lint" in tool.description.lower()
    assert "tier" in tool.args
    assert "test_target" in tool.args


def test_verification_toolset_success():
    toolset = VerificationToolset(root_dir=".", max_attempts=3)

    # Mock subprocess runs to return success (rc=0)
    with patch.object(toolset, "_run_subprocess", return_value=(0, "All tests passed", "")):
        report = toolset.run_verification(tier="all")
        assert "PASSED" in report
        assert toolset.last_passed is True
        assert toolset.attempt_count == 0


def test_verification_toolset_failure_and_escalation():
    toolset = VerificationToolset(root_dir=".", max_attempts=3)

    # Mock subprocess run to simulate a test failure
    with patch.object(toolset, "_run_subprocess", return_value=(1, "", "AssertionError: expected True, got False")):
        # Attempt 1
        rep1 = toolset.run_verification(tier="test")
        assert "FAILED" in rep1
        assert "AssertionError" in rep1
        assert toolset.attempt_count == 1
        assert "ESCALATION WARNING" not in rep1

        # Attempt 2
        rep2 = toolset.run_verification(tier="test")
        assert toolset.attempt_count == 2
        assert "ESCALATION WARNING" not in rep2

        # Attempt 3 (hits max_attempts limit)
        rep3 = toolset.run_verification(tier="test")
        assert toolset.attempt_count == 3
        assert "ESCALATION WARNING" in rep3
        assert "Human-in-the-Loop" in rep3


def test_verification_toolset_tier_selection():
    toolset = VerificationToolset(root_dir=".")

    with patch.object(toolset, "_run_subprocess", return_value=(0, "OK", "")) as mock_cmd:
        toolset.run_verification(tier="lint")
        # Should execute lint command
        assert any("ruff" in str(call_args) for call_args in mock_cmd.call_args_list)

    with patch.object(toolset, "_run_subprocess", return_value=(0, "OK", "")) as mock_cmd:
        toolset.run_verification(tier="test", test_target="tests/test_agent_state.py")
        # Should include test target
        called_args = [str(arg) for call_args in mock_cmd.call_args_list for arg in call_args[0][0]]
        assert "tests/test_agent_state.py" in called_args


def test_verification_toolset_failures_log(tmp_path):
    failures_file = tmp_path / "failures.md"
    toolset = VerificationToolset(root_dir=str(tmp_path), failures_path=str(failures_file))

    with patch.object(toolset, "_run_subprocess", return_value=(1, "", "SyntaxError in main.py")):
        report = toolset.run_verification(tier="lint")
        assert "FAILED" in report
        assert failures_file.exists()
        content = failures_file.read_text(encoding="utf-8")
        assert "verification" in content
        assert "SyntaxError in main.py" in content
        assert "Recent Failures Log" in report

