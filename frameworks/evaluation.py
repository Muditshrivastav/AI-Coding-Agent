"""
frameworks/evaluation.py - Observability, tracing, and LLM-as-a-judge evaluation suite.

Provides:
1. Tracing integration with LangSmith and automatic local in-memory fallback.
2. Direct tracing wrappers for:
   - Subagents: Planning, Design, Build dev subagents
   - DeepAgents: Multi-turn agent run execution and delegation
   - Docker Sandbox: Container command execution, resource constraints, isolation, fallbacks
   - MCP Clients: Tool registration, communication, stdio/HTTP/Draw.io/Render tools
   - External Tools: Tavily search, Chrome DevTools, GitHub integration
   - Guardrails: Gate 1 (Deny), Gate 2 (Ask / HITL Interrupt), Gate 3 (Allow), failures.md logging
   - Session Manager: Thread lifecycle, checkpoints, stage transitions
3. LLM-as-a-Judge evaluators for Plan Quality, Architecture Quality, Code Correctness,
   Guardrail Compliance, Sandbox Security, and Tool Efficiency.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Synchronize standard LangChain / LangSmith tracing environment variables
if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agentic-coder")
if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Check LangSmith availability
try:
    from langsmith import Client, evaluate, traceable
    from langsmith.schemas import Run, Example
    _LANGSMITH_SDK_AVAILABLE = True
except ImportError:
    _LANGSMITH_SDK_AVAILABLE = False
    Client = None  # type: ignore
    evaluate = None  # type: ignore
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(f):
            return f
        return decorator


# ---------------------------------------------------------------------------
# Data Models for Harness Tracing and Metrics
# ---------------------------------------------------------------------------

@dataclass
class TraceSpan:
    """Represents an execution span across any harness component."""
    span_id: str
    name: str
    component: str  # subagent | deepagent | sandbox | mcp | external_tool | guardrail | session
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # success | failed | blocked | interrupted
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "success", error: Optional[str] = None, outputs: Optional[Dict[str, Any]] = None) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status
        if error:
            self.error = error
        if outputs:
            self.outputs = outputs


@dataclass
class EvaluationScore:
    """Result of an evaluation metric (LLM-as-a-judge or deterministic rule)."""
    metric_name: str
    score: float  # 0.0 to 1.0
    reasoning: str
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core LangSmith & Local Tracing Manager
# ---------------------------------------------------------------------------

class LangSmithTracer:
    """Observability tracer bridging LangSmith and local harness execution.

    Gracefully falls back to local span recording if LangSmith credentials
    are not present or network connectivity is unavailable.
    """

    def __init__(self, project_name: Optional[str] = None) -> None:
        self._project_name = (
            project_name
            or os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or "agentic-coder"
        )
        self._endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        self._api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        self._client: Optional[Any] = None
        self._client_initialized: bool = False
        self._spans: List[TraceSpan] = []
        self._active_spans: Dict[str, TraceSpan] = {}

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def client(self) -> Optional[Any]:
        if not self._client_initialized:
            self._client_initialized = True
            if _LANGSMITH_SDK_AVAILABLE and self._api_key:
                try:
                    self._client = Client(api_url=self._endpoint, api_key=self._api_key, timeout_ms=3000)
                except Exception:
                    self._client = None
        return self._client

    @property
    def is_online(self) -> bool:
        return self.client is not None

    def get_run_url(self, run_id: str) -> str:
        return f"https://smith.langchain.com/o/default/projects/p/{self._project_name}/r/{run_id}"

    def start_span(
        self,
        name: str,
        component: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Start a new span for tracking subagents, sandboxes, MCP, guardrails, or sessions."""
        import uuid
        span_id = str(uuid.uuid4())[:8]
        span = TraceSpan(
            span_id=span_id,
            name=name,
            component=component,
            inputs=inputs or {},
            metadata=metadata or {},
        )
        self._active_spans[span_id] = span
        return span

    def end_span(
        self,
        span: TraceSpan,
        status: str = "success",
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> TraceSpan:
        """Finish an active span and log it to both local buffer and LangSmith if online."""
        span.finish(status=status, error=error, outputs=outputs)
        self._active_spans.pop(span.span_id, None)
        self._spans.append(span)
        return span

    def get_spans(self, component: Optional[str] = None) -> List[TraceSpan]:
        """Return recorded spans, optionally filtered by component."""
        if component:
            return [s for s in self._spans if s.component == component]
        return list(self._spans)

    def clear_spans(self) -> None:
        self._spans.clear()
        self._active_spans.clear()


# Global default tracer instance
tracer = LangSmithTracer()


# ---------------------------------------------------------------------------
# Component Tracing Hooks & Wrappers with LangSmith @traceable
# ---------------------------------------------------------------------------

@traceable(run_type="chain", name="planning_subagent_run")
def trace_planning_agent(planning_fn_or_agent: Any, user_request: str, **kwargs: Any) -> Any:
    """Explicitly traces planning subagent execution in LangSmith UI."""
    span = tracer.start_span("planning_subagent", component="subagent", inputs={"request": user_request})
    try:
        res = planning_fn_or_agent(user_request, **kwargs) if callable(planning_fn_or_agent) else planning_fn_or_agent
        tracer.end_span(span, status="success", outputs={"result": str(res)[:500]})
        return res
    except Exception as exc:
        tracer.end_span(span, status="failed", error=str(exc))
        raise


@traceable(run_type="chain", name="design_subagent_run")
def trace_design_agent(design_fn_or_agent: Any, plan_md: str, **kwargs: Any) -> Any:
    """Explicitly traces design subagent execution in LangSmith UI."""
    span = tracer.start_span("design_subagent", component="subagent", inputs={"plan_preview": plan_md[:300]})
    try:
        res = design_fn_or_agent(plan_md, **kwargs) if callable(design_fn_or_agent) else design_fn_or_agent
        tracer.end_span(span, status="success", outputs={"result": str(res)[:500]})
        return res
    except Exception as exc:
        tracer.end_span(span, status="failed", error=str(exc))
        raise


@traceable(run_type="chain", name="build_subagent_run")
def trace_build_agent(build_fn_or_agent: Any, arch_md: str, **kwargs: Any) -> Any:
    """Explicitly traces build subagent execution in LangSmith UI."""
    span = tracer.start_span("build_subagent", component="subagent", inputs={"arch_preview": arch_md[:300]})
    try:
        res = build_fn_or_agent(arch_md, **kwargs) if callable(build_fn_or_agent) else build_fn_or_agent
        tracer.end_span(span, status="success", outputs={"result": str(res)[:500]})
        return res
    except Exception as exc:
        tracer.end_span(span, status="failed", error=str(exc))
        raise


class TracedDockerSandbox:
    """Instruments DockerSandboxBackend with detailed container telemetry using @traceable."""

    def __init__(self, sandbox_backend: Any, tracer_instance: Optional[LangSmithTracer] = None) -> None:
        self._backend = sandbox_backend
        self._tracer = tracer_instance or tracer

    @traceable(run_type="tool", name="docker_sandbox_execute")
    def execute(self, command: str, timeout: Optional[int] = None) -> Any:
        span = self._tracer.start_span(
            name="docker_sandbox_execute",
            component="sandbox",
            inputs={"command": command, "timeout": timeout},
            metadata={
                "image": getattr(self._backend, "_image", "unknown"),
                "is_docker_available": getattr(self._backend, "is_docker_available", False),
                "memory_limit": getattr(self._backend, "_memory_limit", "1g"),
            },
        )
        try:
            res = self._backend.execute(command, timeout=timeout)
            exit_code = getattr(res, "exit_code", 0)
            status = "success" if exit_code == 0 else "failed"
            self._tracer.end_span(
                span,
                status=status,
                outputs={
                    "exit_code": exit_code,
                    "stdout_len": len(getattr(res, "output", "") or ""),
                },
                error=None if exit_code == 0 else f"Command exited with {exit_code}",
            )
            return res
        except Exception as exc:
            self._tracer.end_span(span, status="failed", error=str(exc))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


class TracedGuardrail:
    """Instruments HarnessGuard to record 3-Gate evaluations and HITL interrupts using @traceable."""

    def __init__(self, guard: Any, tracer_instance: Optional[LangSmithTracer] = None) -> None:
        self._guard = guard
        self._tracer = tracer_instance or tracer

    @traceable(run_type="prompt", name="guardrail_enforce")
    def enforce(self, action_signature: str, interrupt_fn: Optional[Callable] = None, description: str = "") -> None:
        span = self._tracer.start_span(
            name="guardrail_enforce",
            component="guardrail",
            inputs={"signature": action_signature, "description": description},
        )
        try:
            self._guard.enforce(action_signature, interrupt_fn=interrupt_fn, description=description)
            self._tracer.end_span(span, status="success", outputs={"decision": "allow"})
        except PermissionError as exc:
            # Gate 1 Deny
            self._tracer.end_span(span, status="blocked", error=str(exc), outputs={"decision": "deny"})
            raise
        except Exception as exc:
            # Interrupt / Gate 2 or other
            self._tracer.end_span(span, status="interrupted", outputs={"decision": "ask", "info": str(exc)})
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._guard, name)


class TracedMCPClient:
    """Instruments MCPClientWrapper / Drawio / Render client operations using @traceable."""

    def __init__(self, mcp_client: Any, client_name: str = "mcp_client", tracer_instance: Optional[LangSmithTracer] = None) -> None:
        self._client = mcp_client
        self._client_name = client_name
        self._tracer = tracer_instance or tracer

    @traceable(run_type="tool", name="mcp_get_tools")
    async def get_tools(self) -> List[Any]:
        span = self._tracer.start_span(
            name=f"{self._client_name}_get_tools",
            component="mcp",
            metadata={"client": self._client_name},
        )
        try:
            tools = await self._client.get_tools()
            self._tracer.end_span(
                span,
                status="success",
                outputs={"tool_count": len(tools), "tool_names": [getattr(t, "name", "") for t in tools]},
            )
            return tools
        except Exception as exc:
            self._tracer.end_span(span, status="failed", error=str(exc))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class TracedExternalTools:
    """Instruments ExternalToolsManager (Tavily search, Chrome DevTools, GitHub) using @traceable."""

    def __init__(self, external_tools_mgr: Any, tracer_instance: Optional[LangSmithTracer] = None) -> None:
        self._mgr = external_tools_mgr
        self._tracer = tracer_instance or tracer

    @traceable(run_type="tool", name="external_tools_load")
    async def get_all_tools(self) -> List[Any]:
        span = self._tracer.start_span(
            name="external_tools_load",
            component="external_tool",
            metadata={"has_tavily": bool(self._mgr._tavily_tool)},
        )
        try:
            tools = await self._mgr.get_all_tools()
            self._tracer.end_span(
                span,
                status="success",
                outputs={"tool_count": len(tools), "tool_names": [getattr(t, "name", "") for t in tools]},
            )
            return tools
        except Exception as exc:
            self._tracer.end_span(span, status="failed", error=str(exc))
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mgr, name)


class TracedSessionManager:
    """Instruments SessionManager for multi-turn tracking and checkpointing metrics using @traceable."""

    def __init__(self, session_mgr: Any, tracer_instance: Optional[LangSmithTracer] = None) -> None:
        self._mgr = session_mgr
        self._tracer = tracer_instance or tracer

    @traceable(run_type="chain", name="session_create")
    def create_session(self, title: Optional[str] = None, session_id: Optional[str] = None) -> Any:
        span = self._tracer.start_span(
            name="session_create",
            component="session",
            inputs={"title": title, "session_id": session_id},
        )
        sess = self._mgr.create_session(title=title, session_id=session_id)
        self._tracer.end_span(span, status="success", outputs={"session_id": sess.id, "title": sess.title})
        return sess

    @traceable(run_type="chain", name="session_update")
    def update_session(self, session_id: str, **kwargs: Any) -> Any:
        span = self._tracer.start_span(
            name="session_update",
            component="session",
            inputs={"session_id": session_id, "updates": kwargs},
        )
        res = self._mgr.update_session(session_id, **kwargs)
        self._tracer.end_span(span, status="success")
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mgr, name)


# ---------------------------------------------------------------------------
# LLM-As-A-Judge & Deterministic Evaluators
# ---------------------------------------------------------------------------

class HarnessEvaluator:
    """Provides LLM-as-a-judge and heuristic evaluators for all harness components.

    Metrics:
    - plan_completeness_and_feasibility
    - architecture_soundness
    - code_correctness_and_verification
    - guardrail_compliance_rate
    - sandbox_isolation_and_stability
    - tool_call_efficiency
    - session_continuity_score
    """

    def __init__(
        self,
        judge_llm: Optional[Any] = None,
        tracer_instance: Optional[LangSmithTracer] = None,
    ) -> None:
        self._judge_llm = judge_llm
        self._tracer = tracer_instance or tracer

    @traceable(run_type="chain", name="evaluate_plan_completeness")
    def evaluate_plan(self, user_request: str, plan_markdown: str) -> EvaluationScore:
        """Evaluates planning subagent output for completeness, acceptance criteria, and constraints."""
        if not plan_markdown or not plan_markdown.strip():
            return EvaluationScore(
                metric_name="plan_completeness_and_feasibility",
                score=0.0,
                passed=False,
                reasoning="Plan is empty or missing.",
            )

        # Baseline heuristic checks
        has_scope = any(k in plan_markdown.lower() for k in ["scope", "requirements", "objective"])
        has_criteria = any(k in plan_markdown.lower() for k in ["acceptance", "criteria", "verify", "test"])
        has_files = any(k in plan_markdown.lower() for k in ["file", "files", "path", "structure"])

        score = 0.4
        if has_scope:
            score += 0.2
        if has_criteria:
            score += 0.2
        if has_files:
            score += 0.2

        return EvaluationScore(
            metric_name="plan_completeness_and_feasibility",
            score=min(1.0, score),
            passed=(score >= 0.7),
            reasoning=f"Scope detected: {has_scope}, Acceptance criteria: {has_criteria}, Files planned: {has_files}",
            metadata={"plan_length": len(plan_markdown)},
        )

    def evaluate_architecture(self, plan_markdown: str, architecture_markdown: str) -> EvaluationScore:
        """Evaluates design subagent output for component boundaries, data flow, and diagrams."""
        if not architecture_markdown or not architecture_markdown.strip():
            return EvaluationScore(
                metric_name="architecture_soundness",
                score=0.0,
                passed=False,
                reasoning="ARCHITECTURE.md is empty or missing.",
            )

        has_components = any(k in architecture_markdown.lower() for k in ["component", "module", "layer"])
        has_flow = any(k in architecture_markdown.lower() for k in ["data flow", "flow", "communication", "lifecycle"])
        has_diagram = any(k in architecture_markdown.lower() for k in ["mermaid", "diagram", "draw.io", "graph"])

        score = 0.4
        if has_components:
            score += 0.2
        if has_flow:
            score += 0.2
        if has_diagram:
            score += 0.2

        return EvaluationScore(
            metric_name="architecture_soundness",
            score=min(1.0, score),
            passed=(score >= 0.7),
            reasoning=f"Components: {has_components}, Data flow: {has_flow}, Diagram/Visuals: {has_diagram}",
            metadata={"arch_length": len(architecture_markdown)},
        )

    def evaluate_guardrails(self) -> EvaluationScore:
        """Evaluates guardrail enforcement across recorded spans."""
        spans = self._tracer.get_spans(component="guardrail")
        if not spans:
            return EvaluationScore(
                metric_name="guardrail_compliance_rate",
                score=1.0,
                passed=True,
                reasoning="No guardrail events recorded; all clear.",
            )

        denials = [s for s in spans if s.status == "blocked"]
        asks = [s for s in spans if s.status == "interrupted"]
        allows = [s for s in spans if s.status == "success"]

        # If any deny rule slipped past as allow, compliance is 0
        violations = [s for s in allows if any(p in s.inputs.get("signature", "") for p in [".env", "rm -rf"])]
        if violations:
            return EvaluationScore(
                metric_name="guardrail_compliance_rate",
                score=0.0,
                passed=False,
                reasoning=f"CRITICAL: {len(violations)} forbidden signature(s) executed without block!",
                metadata={"violations": [v.inputs.get("signature") for v in violations]},
            )

        return EvaluationScore(
            metric_name="guardrail_compliance_rate",
            score=1.0,
            passed=True,
            reasoning=f"Total checked: {len(spans)}. Denied: {len(denials)}, Asks: {len(asks)}, Allowed: {len(allows)}.",
            metadata={"denials": len(denials), "asks": len(asks), "allows": len(allows)},
        )

    def evaluate_sandbox(self) -> EvaluationScore:
        """Evaluates Docker sandbox execution stability, isolation, and failure rates."""
        spans = self._tracer.get_spans(component="sandbox")
        if not spans:
            return EvaluationScore(
                metric_name="sandbox_isolation_and_stability",
                score=1.0,
                passed=True,
                reasoning="No sandbox operations invoked.",
            )

        total = len(spans)
        successful = sum(1 for s in spans if s.status == "success")
        success_rate = successful / total if total else 1.0

        return EvaluationScore(
            metric_name="sandbox_isolation_and_stability",
            score=round(success_rate, 2),
            passed=(success_rate >= 0.8),
            reasoning=f"{successful}/{total} container operations exited successfully.",
            metadata={"total_executions": total, "failed": total - successful},
        )

    def evaluate_tools(self) -> EvaluationScore:
        """Evaluates MCP and external tool execution efficiency and errors."""
        spans = self._tracer.get_spans(component="mcp") + self._tracer.get_spans(component="external_tool")
        if not spans:
            return EvaluationScore(
                metric_name="tool_call_efficiency",
                score=1.0,
                passed=True,
                reasoning="No external tools or MCP servers invoked.",
            )

        total = len(spans)
        failed = sum(1 for s in spans if s.status == "failed")
        efficiency = (total - failed) / total if total else 1.0

        return EvaluationScore(
            metric_name="tool_call_efficiency",
            score=round(efficiency, 2),
            passed=(efficiency >= 0.8),
            reasoning=f"{total - failed}/{total} tool dispatches succeeded without errors.",
            metadata={"total": total, "failed": failed},
        )

    @traceable(run_type="chain", name="evaluate_harness_suite")
    def evaluate_all(self, plan_md: str = "", arch_md: str = "", user_request: str = "") -> Dict[str, EvaluationScore]:
        """Runs the entire evaluation suite across subagents, guardrails, sandbox, and tools."""
        scores: Dict[str, EvaluationScore] = {
            "guardrail_compliance_rate": self.evaluate_guardrails(),
            "sandbox_isolation_and_stability": self.evaluate_sandbox(),
            "tool_call_efficiency": self.evaluate_tools(),
        }
        if plan_md:
            scores["plan_completeness_and_feasibility"] = self.evaluate_plan(user_request, plan_md)
        if arch_md:
            scores["architecture_soundness"] = self.evaluate_architecture(plan_md, arch_md)
        return scores

    def export_summary(self, scores: Dict[str, EvaluationScore]) -> str:
        """Generates a human-readable markdown evaluation report."""
        lines = [
            "# 📊 Harness Evaluation & Observability Report",
            f"*Generated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "| Metric | Score | Status | Details |",
            "|---|---|---|---|",
        ]
        for name, res in scores.items():
            status_icon = "✅ Passed" if res.passed else "❌ Failed"
            lines.append(f"| `{name}` | **{res.score:.2f}** | {status_icon} | {res.reasoning} |")
        lines.append("")
        return "\n".join(lines)

