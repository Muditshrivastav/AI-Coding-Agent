"""
tests/test_evaluation.py - Unit tests for the evaluation and tracing suite in frameworks/evaluation.py.
"""

import pytest
from unittest.mock import MagicMock
from frameworks.evaluation import (
    LangSmithTracer,
    TraceSpan,
    TracedDockerSandbox,
    TracedGuardrail,
    TracedMCPClient,
    TracedExternalTools,
    TracedSessionManager,
    HarnessEvaluator,
    EvaluationScore,
)


def test_tracer_span_lifecycle():
    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()
    
    span = tracer.start_span("planning_subagent", component="subagent", inputs={"prompt": "build api"})
    assert span.name == "planning_subagent"
    assert span.component == "subagent"
    assert span.status == "success"
    
    tracer.end_span(span, status="success", outputs={"plan": "PLAN.md generated"})
    assert span.duration_ms >= 0.0
    assert span.outputs["plan"] == "PLAN.md generated"
    
    spans = tracer.get_spans("subagent")
    assert len(spans) == 1
    assert spans[0].name == "planning_subagent"


def test_traced_docker_sandbox():
    mock_backend = MagicMock()
    mock_resp = MagicMock()
    mock_resp.exit_code = 0
    mock_resp.output = "test container output"
    mock_backend.execute.return_value = mock_resp
    mock_backend._image = "ai-coding-agent-sandbox:latest"
    mock_backend.is_docker_available = True

    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()
    traced_sandbox = TracedDockerSandbox(mock_backend, tracer_instance=tracer)

    res = traced_sandbox.execute("pytest -q")
    assert res.exit_code == 0
    
    spans = tracer.get_spans(component="sandbox")
    assert len(spans) == 1
    assert spans[0].name == "docker_sandbox_execute"
    assert spans[0].outputs["exit_code"] == 0
    assert spans[0].status == "success"


def test_traced_guardrail_enforce():
    mock_guard = MagicMock()
    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()
    traced_guard = TracedGuardrail(mock_guard, tracer_instance=tracer)

    # Allowed signature
    traced_guard.enforce("Bash(pytest)")
    spans = tracer.get_spans(component="guardrail")
    assert len(spans) == 1
    assert spans[0].status == "success"
    assert spans[0].outputs["decision"] == "allow"

    # Blocked signature (Gate 1 Deny)
    mock_guard.enforce.side_effect = PermissionError("Blocked pattern")
    with pytest.raises(PermissionError):
        traced_guard.enforce("Bash(rm -rf /)")
    
    spans = tracer.get_spans(component="guardrail")
    assert len(spans) == 2
    assert spans[1].status == "blocked"
    assert spans[1].outputs["decision"] == "deny"


def test_traced_mcp_client():
    import asyncio
    mock_mcp = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "github_create_issue"
    
    async def fake_get_tools():
        return [mock_tool]
        
    mock_mcp.get_tools = fake_get_tools
    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()

    traced_client = TracedMCPClient(mock_mcp, client_name="github_mcp", tracer_instance=tracer)
    tools = asyncio.run(traced_client.get_tools())
    assert len(tools) == 1
    
    spans = tracer.get_spans(component="mcp")
    assert len(spans) == 1
    assert spans[0].outputs["tool_names"] == ["github_create_issue"]


def test_traced_session_manager():
    mock_mgr = MagicMock()
    mock_session = MagicMock()
    mock_session.id = "session-123"
    mock_session.title = "Test Session"
    mock_mgr.create_session.return_value = mock_session

    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()
    traced_mgr = TracedSessionManager(mock_mgr, tracer_instance=tracer)

    sess = traced_mgr.create_session(title="Test Session")
    assert sess.id == "session-123"
    
    spans = tracer.get_spans(component="session")
    assert len(spans) == 1
    assert spans[0].name == "session_create"


def test_evaluator_metrics_and_summary():
    tracer = LangSmithTracer(project_name="test-project")
    tracer.clear_spans()
    evaluator = HarnessEvaluator(tracer_instance=tracer)

    plan_text = """
    # PLAN
    ## Scope and Requirements
    Build auth endpoints.
    ## Acceptance Criteria
    - Returns 200 on login
    ## File Structure
    - src/auth.py
    """
    plan_score = evaluator.evaluate_plan("Build auth", plan_text)
    assert plan_score.passed is True
    assert plan_score.score >= 0.8

    arch_text = """
    # ARCHITECTURE
    ## Components
    FastAPI controller and DB service.
    ## Data Flow
    Client -> Route -> Service -> DB.
    ## Mermaid Diagram
    ```mermaid
    graph TD;
    A-->B;
    ```
    """
    arch_score = evaluator.evaluate_architecture(plan_text, arch_text)
    assert arch_score.passed is True
    assert arch_score.score >= 0.8

    scores = evaluator.evaluate_all(plan_md=plan_text, arch_md=arch_text, user_request="Build auth")
    assert "guardrail_compliance_rate" in scores
    assert "sandbox_isolation_and_stability" in scores
    assert "tool_call_efficiency" in scores
    assert "plan_completeness_and_feasibility" in scores
    assert "architecture_soundness" in scores

    summary = evaluator.export_summary(scores)
    assert "Harness Evaluation & Observability Report" in summary
    assert "plan_completeness_and_feasibility" in summary
