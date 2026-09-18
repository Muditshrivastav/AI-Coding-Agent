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


# ---------------------------------------------------------------------------
# LangGraph StateGraph Unit Tests
# ---------------------------------------------------------------------------

from tools.verification_tools import _LANGGRAPH_AVAILABLE

pytestmark_langgraph = pytest.mark.skipif(
    not _LANGGRAPH_AVAILABLE,
    reason="langgraph is not installed in current Python environment",
)


@pytestmark_langgraph
def test_verification_graph_compilation():
    """Verify that build_graph compiles a runnable LangGraph StateGraph."""
    from tools.verification_tools import build_verification_graph

    graph = build_verification_graph(root_dir=".")
    assert graph is not None
    # Check that expected node keys are in the compiled graph
    assert "lint_node" in graph.nodes
    assert "test_node" in graph.nodes
    assert "aggregate_node" in graph.nodes
    assert "pass_node" in graph.nodes
    assert "hitl_escalation_node" in graph.nodes


@pytestmark_langgraph
def test_verification_graph_success_flow():
    """Verify the full graph execution path when all checks pass."""
    toolset = VerificationToolset(root_dir=".", max_attempts=3)

    with patch.object(toolset, "_run_subprocess", return_value=(0, "Clean run", "")):
        graph = toolset.build_graph()
        result = graph.invoke({"tier": "all"})

        assert result["verify_passed"] is True
        assert result["verify_attempts"] == 0
        assert "PASSED" in result["diagnostic_report"]
        assert result["escalation_payload"] is None
        assert result["lint_result"]["returncode"] == 0
        assert result["test_result"]["returncode"] == 0


@pytestmark_langgraph
def test_verification_graph_failure_and_retry_flow():
    """Verify failure path without breach: attempts incremented and no escalation."""
    toolset = VerificationToolset(root_dir=".", max_attempts=3)

    with patch.object(toolset, "_run_subprocess", return_value=(1, "", "ruff check failed")):
        graph = toolset.build_graph()
        # First failure attempt
        res1 = graph.invoke({"tier": "lint", "verify_attempts": 0})
        assert res1["verify_passed"] is False
        assert res1["verify_attempts"] == 1
        assert res1["escalation_payload"] is None
        assert "FAILED" in res1["diagnostic_report"]


@pytestmark_langgraph
def test_verification_graph_hitl_escalation():
    """Verify that reaching max_attempts routes to hitl_escalation_node with structured payload."""
    escalation_interrupted = []

    def mock_interrupt(payload):
        escalation_interrupted.append(payload)

    toolset = VerificationToolset(
        root_dir=".",
        max_attempts=2,
        interrupt_fn=mock_interrupt,
    )

    with patch.object(toolset, "_run_subprocess", return_value=(1, "", "pytest error")):
        graph = toolset.build_graph()

        # Invoking with verify_attempts=1 means this execution will be attempt #2 (reaches max_attempts=2)
        res = graph.invoke({"tier": "test", "verify_attempts": 1})

        assert res["verify_passed"] is False
        assert res["verify_attempts"] == 2
        assert res["escalation_payload"] is not None
        assert res["escalation_payload"]["action"] == "verification_escalation"
        assert res["escalation_payload"]["attempts"] == 2
        assert len(escalation_interrupted) == 1
        assert escalation_interrupted[0]["action"] == "verification_escalation"


@pytestmark_langgraph
def test_verification_graph_tier_skipping():
    """Verify that specifying tier='lint' skips pytest execution, and tier='test' skips linting."""
    toolset = VerificationToolset(root_dir=".")

    # Tier: lint only
    with patch.object(toolset, "run_lint", return_value={"step": "ruff check", "returncode": 0, "stdout": "ok", "stderr": ""}) as mock_lint, \
         patch.object(toolset, "run_tests") as mock_test:
        graph = toolset.build_graph()
        res = graph.invoke({"tier": "lint"})
        assert mock_lint.called
        assert not mock_test.called
        assert res["test_result"]["stdout"] == "(skipped)"

    # Tier: test only
    with patch.object(toolset, "run_lint") as mock_lint, \
         patch.object(toolset, "run_tests", return_value={"step": "pytest", "returncode": 0, "stdout": "ok", "stderr": ""}) as mock_test:
        graph = toolset.build_graph()
        res = graph.invoke({"tier": "test", "test_target": "tests/test_agent_state.py"})
        assert not mock_lint.called
        assert mock_test.called
        assert res["lint_result"]["stdout"] == "(skipped)"

