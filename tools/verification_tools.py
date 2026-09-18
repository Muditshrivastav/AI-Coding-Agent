"""
tools/verification_tools.py - Self-verification toolset for code validation and repair loops.
Executes linting, static checks, and unit tests, capturing structured failure tracebacks.

LangGraph StateGraph
--------------------
In addition to the StructuredTool API (run_verification / get_tools), this module
exposes a compiled LangGraph StateGraph via ``build_verification_graph()``.

Graph topology::

    START
      └─► lint_node ──► test_node ──► aggregate_node
                                            │
                        ┌───────────────────┴───────────────────┐
                        │ all_passed = True                      │ all_passed = False
                        ▼                                        ▼
                   pass_node ──► END            attempts < max_attempts?
                                                  Yes ──► END  (caller retries node)
                                                  No  ──► hitl_escalation_node ──► END

Each node is registered with a ``RetryPolicy`` that retries only on transient
infrastructure exceptions (subprocess timeout / command not found), NOT on
logic failures (non-zero exit codes). Logic-failure retries are counted via
``verify_attempts`` in the graph state.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import shutil
import subprocess
from typing import Any, Callable

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ---------------------------------------------------------------------------
# LangGraph imports — guarded so the module remains importable even if
# langgraph is not installed (StructuredTool path still works fine).
# ---------------------------------------------------------------------------
try:
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from langgraph.types import RetryPolicy
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pydantic schema for the StructuredTool interface (unchanged)
# ---------------------------------------------------------------------------

class VerificationArgs(BaseModel):
    tier: str = Field(
        default="all",
        description="Verification tier to run: 'all' (lint + test), 'lint' (ruff static checks), or 'test' (pytest suite).",
    )
    test_target: str | None = Field(
        default=None,
        description="Optional specific test file or path to run (e.g. 'tests/test_build_subagent.py').",
    )


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

if _LANGGRAPH_AVAILABLE:
    class VerificationGraphState(TypedDict, total=False):
        """Typed state flowing through the verification StateGraph.

        Fields
        ------
        tier            : Verification tier — ``"all"`` | ``"lint"`` | ``"test"``.
        test_target     : Optional path to a specific pytest target.
        lint_result     : Structured result dict from the lint stage.
        test_result     : Structured result dict from the test stage.
        verify_attempts : Running count of consecutive failed attempts.
        verify_passed   : True only when every stage in the current run passed.
        escalation_payload : Payload forwarded to the HITL interrupt node on breach.
        diagnostic_report  : Human-readable diagnostics string built by aggregate_node.
        """
        tier: str
        test_target: str | None
        lint_result: dict[str, Any]
        test_result: dict[str, Any]
        verify_attempts: int
        verify_passed: bool
        escalation_payload: dict[str, Any] | None
        diagnostic_report: str


# ---------------------------------------------------------------------------
# Retry predicate — only retry on transient infrastructure errors
# ---------------------------------------------------------------------------

def _is_transient_error(exc: Exception) -> bool:
    """Return True for transient subprocess failures that warrant a node retry.

    LangGraph's RetryPolicy calls this predicate with the raised exception.
    We retry only on OS-level failures (timeout, missing executable) — never on
    logic failures (non-zero exit code from ruff / pytest).
    """
    return isinstance(exc, (subprocess.TimeoutExpired, FileNotFoundError, OSError))


# ---------------------------------------------------------------------------
# VerificationToolset
# ---------------------------------------------------------------------------

class VerificationToolset:
    """Manages programmatic self-verification execution, failure logging to harness/failures.md, and error feedback.

    Exposes two interfaces:
    1. ``get_tools()``           — LangChain StructuredTool list (consumed by build-agent).
    2. ``build_graph()``        — Compiled LangGraph ``StateGraph`` with RetryPolicy,
                                   conditional edges, and HITL escalation node.
    """

    def __init__(
        self,
        root_dir: str = ".",
        max_attempts: int = 3,
        failures_path: str | None = None,
        interrupt_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._max_attempts = max_attempts
        self._failures_path = failures_path or os.path.join(root_dir, "harness", "failures.md")
        self._attempt_count: int = 0
        self._last_passed: bool = False
        # Optional LangGraph interrupt callable — when set, a real HITL checkpoint
        # is created after max_attempts consecutive failures instead of a text warning.
        self._interrupt_fn: Callable[[Any], Any] | None = interrupt_fn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def attempt_count(self) -> int:
        """Current number of consecutive verification attempts for the active todo/task.

        Increments by 1 on each call to run_verification(). Resets to 0 upon any passing run
        or explicit call to reset_attempts().
        """
        return self._attempt_count

    @property
    def max_attempts(self) -> int:
        """Maximum allowed failure retries (default: 3) before escalating to Human-in-the-Loop."""
        return self._max_attempts

    @property
    def last_passed(self) -> bool:
        """Boolean flag indicating whether the most recent verification execution succeeded (exit code 0)."""
        return self._last_passed

    def is_escalation_required(self) -> bool:
        """Returns True if consecutive failures reached or exceeded max_attempts.

        Used by harness guards and orchestrators to gate execution and trigger HITL approval.
        """
        return self._attempt_count >= self._max_attempts

    def reset_attempts(self) -> None:
        """Resets attempt counter to 0 and clears last_passed flag.

        Call this when switching to a new todo item or after human intervention fixes an error.
        """
        self._attempt_count = 0
        self._last_passed = False

    def sync_to_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Syncs verification metrics directly into LangGraph AgentState dictionary."""
        state["verify_attempts"] = self._attempt_count
        state["verify_passed"] = self._last_passed
        if self._last_passed:
            state["current_stage"] = "verify"
        return state

    # ------------------------------------------------------------------
    # Internal subprocess helper
    # ------------------------------------------------------------------

    def _run_subprocess(self, cmd: list[str]) -> tuple[int, str, str]:
        """Runs a command inside root_dir, returning (returncode, stdout, stderr).

        Raises ``subprocess.TimeoutExpired`` and ``FileNotFoundError`` deliberately
        so that ``RetryPolicy`` can intercept transient infrastructure errors.
        All other exceptions are caught and returned as (1, "", message).
        """
        res = subprocess.run(
            cmd,
            cwd=self._root_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return res.returncode, res.stdout, res.stderr

    # ------------------------------------------------------------------
    # Public stage methods  (used directly by graph nodes)
    # ------------------------------------------------------------------

    def run_lint(self, path: str = ".") -> dict[str, Any]:
        """Run ruff static analysis and return a structured result dict.

        Falls back to ``python -m ruff`` if the ``ruff`` binary is not on PATH.
        Raises ``subprocess.TimeoutExpired`` / ``FileNotFoundError`` on transient
        OS-level failures so that ``RetryPolicy`` can retry the node.

        Returns
        -------
        dict with keys: ``step``, ``returncode``, ``stdout``, ``stderr``
        """
        rc, out, err = self._run_subprocess(["ruff", "check", path])
        if rc == 127:
            # ruff binary missing — try module invocation
            rc, out, err = self._run_subprocess(["python", "-m", "ruff", "check", path])
        return {"step": "ruff check", "returncode": rc, "stdout": out, "stderr": err}

    def run_tests(self, test_target: str | None = None) -> dict[str, Any]:
        """Run pytest and return a structured result dict.

        Falls back to ``python -m pytest`` if the ``pytest`` binary is not on PATH.
        Raises ``subprocess.TimeoutExpired`` / ``FileNotFoundError`` on transient
        OS-level failures so that ``RetryPolicy`` can retry the node.

        Returns
        -------
        dict with keys: ``step``, ``returncode``, ``stdout``, ``stderr``
        """
        cmd = ["pytest", "-q"]
        if test_target:
            cmd.append(test_target)
        rc, out, err = self._run_subprocess(cmd)
        if rc == 127:
            cmd[0:1] = ["python", "-m", "pytest"]
            rc, out, err = self._run_subprocess(cmd)
        return {"step": "pytest", "returncode": rc, "stdout": out, "stderr": err}

    # ------------------------------------------------------------------
    # Failure logging helpers
    # ------------------------------------------------------------------

    @property
    def failures_path(self) -> str:
        return self._failures_path

    def get_recent_failures(self, max_entries: int = 5) -> list[str]:
        """Reads recent failure entries from harness/failures.md."""
        if not os.path.isfile(self._failures_path):
            return []
        try:
            with open(self._failures_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip().startswith("|") and not line.strip().startswith("| Timestamp") and not line.strip().startswith("|---")]
            return lines[-max_entries:]
        except Exception:
            return []

    def record_failure(
        self,
        description: str,
        stage: str = "build",
        category: str = "verification",
        action_taken: str = "Inspect diagnostics and repair in next turn",
    ) -> None:
        """Appends a categorized failure record to harness/failures.md."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        clean_desc = description.replace("\n", " ").replace("|", "/").strip()
        clean_desc = (clean_desc[:140] + "...") if len(clean_desc) > 140 else clean_desc

        entry = f"| {timestamp} | {stage} | {category} | {clean_desc} | {action_taken} |\n"

        try:
            os.makedirs(os.path.dirname(self._failures_path), exist_ok=True)
            if not os.path.isfile(self._failures_path) or os.path.getsize(self._failures_path) == 0:
                with open(self._failures_path, "w", encoding="utf-8") as f:
                    f.write("# Failure Log and Triage\n\n| Timestamp | Stage | Category | Description | Resolution / Action Taken |\n|---|---|---|---|---|\n")
            with open(self._failures_path, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Flat StructuredTool interface (backward compatible)
    # ------------------------------------------------------------------

    def run_verification(self, tier: str = "all", test_target: str | None = None) -> str:
        """Executes verification checks and formats actionable diagnostics for repair loops."""
        self._attempt_count += 1
        tier = (tier or "all").lower()

        results: list[dict[str, Any]] = []
        has_make = shutil.which("make") is not None

        # Execute based on tier
        if tier in ("all", "lint"):
            if has_make and tier == "all" and not test_target:
                try:
                    rc, out, err = self._run_subprocess(["make", "verify"])
                except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                    rc, out, err = 1, "", str(exc)
                results.append({"step": "make verify", "returncode": rc, "stdout": out, "stderr": err})
            else:
                results.append(self.run_lint())

        if tier in ("all", "test"):
            # Skip if make verify already ran both and succeeded
            if not (has_make and tier == "all" and not test_target and results and results[0]["returncode"] == 0):
                results.append(self.run_tests(test_target))

        all_passed = len(results) > 0 and all(r["returncode"] == 0 for r in results)
        self._last_passed = all_passed

        if all_passed:
            current_attempt = self._attempt_count
            self._attempt_count = 0  # reset count on clean pass
            return (
                f"✅ Verification PASSED (tier: {tier}, attempt: {current_attempt}).\n"
                "All checks and tests succeeded without errors."
            )

        # Build detailed diagnostic report
        failure_blocks: list[str] = []
        for r in results:
            if r["returncode"] != 0:
                output_content = (r["stderr"].strip() or r["stdout"].strip()) or "Process returned non-zero exit status without output."
                failure_blocks.append(f"[{r['step']}] exit={r['returncode']}\n{output_content}")

        # Automatically log failure into harness/failures.md
        failed_summary = "; ".join(
            f"{r['step']}: {((r['stderr'].strip() or r['stdout'].strip()).splitlines() or ['failed'])[-1]}"
            for r in results if r["returncode"] != 0
        )
        self.record_failure(
            description=failed_summary,
            stage="build",
            category="verification",
            action_taken="Inspect diagnostics and repair in next turn",
        )

        recent_history = self.get_recent_failures(max_entries=3)
        history_note = ""
        if recent_history:
            history_note = "\n\nRecent Failures Log (harness/failures.md):\n" + "\n".join(recent_history)

        escalation_note = ""
        if self._attempt_count >= self._max_attempts:
            escalation_note = (
                f"\n\n⚠️ ESCALATION WARNING: Verification has failed {self._attempt_count} times in a row. "
                "Capped retry limit reached. Escalate to Human-in-the-Loop (HITL) if error cannot be resolved."
            )
            # Trigger a real LangGraph HITL checkpoint if an interrupt function is wired.
            if self._interrupt_fn is not None:
                self._interrupt_fn({
                    "action": "verification_escalation",
                    "attempts": self._attempt_count,
                    "max_attempts": self._max_attempts,
                    "last_failure": failed_summary,
                    "message": (
                        f"Verification has failed {self._attempt_count} consecutive times. "
                        "Approve to allow the agent to continue retrying, or reject to abort."
                    ),
                })

        report = (
            f"❌ Verification FAILED (tier: {tier}, attempt: {self._attempt_count}/{self._max_attempts}).\n"
            "Diagnostics:\n" + "\n\n".join(failure_blocks) +
            "\n\nRepair Instruction: Examine the traceback above. Fix the root cause in the affected file(s). "
            "Never delete or alter a test simply to bypass failure."
            f"{history_note}"
            f"{escalation_note}"
        )
        return report

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.run_verification,
                name="run_verification",
                description=(
                    "Executes the self-verification loop ('make verify', ruff linting, and pytest suite) and logs triage records to harness/failures.md. "
                    "Returns detailed stdout/stderr tracebacks on failure for iterative code repair. "
                    "Optionally specify tier ('all', 'lint', 'test') or a specific test_target."
                ),
                args_schema=VerificationArgs,
            )
        ]

    # ------------------------------------------------------------------
    # LangGraph StateGraph — node implementations
    # ------------------------------------------------------------------

    def _node_lint(self, state: "VerificationGraphState") -> "VerificationGraphState":
        """Graph node: run ruff linting.

        Raises ``subprocess.TimeoutExpired`` / ``FileNotFoundError`` on transient
        failures so RetryPolicy can automatically retry this node up to its
        configured ``max_attempts`` before propagating the exception.
        """
        tier = state.get("tier", "all")
        if tier not in ("all", "lint"):
            # Tier does not include lint — emit a neutral pass result
            return {"lint_result": {"step": "ruff check", "returncode": 0, "stdout": "(skipped)", "stderr": ""}}
        result = self.run_lint()
        return {"lint_result": result}

    def _node_test(self, state: "VerificationGraphState") -> "VerificationGraphState":
        """Graph node: run pytest.

        Raises ``subprocess.TimeoutExpired`` / ``FileNotFoundError`` on transient
        failures so RetryPolicy can automatically retry this node.
        """
        tier = state.get("tier", "all")
        if tier not in ("all", "test"):
            # Tier does not include tests — emit a neutral pass result
            return {"test_result": {"step": "pytest", "returncode": 0, "stdout": "(skipped)", "stderr": ""}}
        result = self.run_tests(state.get("test_target"))
        return {"test_result": result}

    def _node_aggregate(self, state: "VerificationGraphState") -> "VerificationGraphState":
        """Graph node: merge lint + test results, update attempt counter, build diagnostic report.

        This node never raises — all subprocess errors have been absorbed by the
        earlier nodes.  It is responsible for:
        - Counting consecutive failures in ``verify_attempts``.
        - Writing a failure row to ``harness/failures.md``.
        - Setting ``verify_passed`` and ``diagnostic_report`` in state.
        - Populating ``escalation_payload`` when the attempt cap is breached.
        """
        lint_r: dict[str, Any] = state.get("lint_result", {"step": "ruff check", "returncode": 0, "stdout": "", "stderr": ""})
        test_r: dict[str, Any] = state.get("test_result", {"step": "pytest", "returncode": 0, "stdout": "", "stderr": ""})
        results = [lint_r, test_r]

        all_passed = all(r["returncode"] == 0 for r in results)

        if all_passed:
            return {
                "verify_passed": True,
                "verify_attempts": 0,  # reset on clean pass
                "escalation_payload": None,
                "diagnostic_report": (
                    f"✅ Verification PASSED.\n"
                    "All checks and tests succeeded without errors."
                ),
            }

        # ── Failure path ────────────────────────────────────────────────────
        attempts: int = (state.get("verify_attempts") or 0) + 1

        failure_blocks: list[str] = []
        for r in results:
            if r["returncode"] != 0:
                output_content = (r["stderr"].strip() or r["stdout"].strip()) or "Process returned non-zero exit status without output."
                failure_blocks.append(f"[{r['step']}] exit={r['returncode']}\n{output_content}")

        failed_summary = "; ".join(
            f"{r['step']}: {((r['stderr'].strip() or r['stdout'].strip()).splitlines() or ['failed'])[-1]}"
            for r in results if r["returncode"] != 0
        )

        # Log to harness/failures.md
        self.record_failure(
            description=failed_summary,
            stage="verify",
            category="verification",
            action_taken="Inspect diagnostics and repair in next turn",
        )

        recent_history = self.get_recent_failures(max_entries=3)
        history_note = ""
        if recent_history:
            history_note = "\n\nRecent Failures Log (harness/failures.md):\n" + "\n".join(recent_history)

        escalation_payload: dict[str, Any] | None = None
        escalation_note = ""
        if attempts >= self._max_attempts:
            escalation_payload = {
                "action": "verification_escalation",
                "attempts": attempts,
                "max_attempts": self._max_attempts,
                "last_failure": failed_summary,
                "message": (
                    f"Verification has failed {attempts} consecutive times. "
                    "Approve to allow the agent to continue retrying, or reject to abort."
                ),
            }
            escalation_note = (
                f"\n\n⚠️ ESCALATION: {attempts}/{self._max_attempts} consecutive failures. "
                "Routing to HITL checkpoint."
            )

        diagnostic_report = (
            f"❌ Verification FAILED (attempt: {attempts}/{self._max_attempts}).\n"
            "Diagnostics:\n" + "\n\n".join(failure_blocks) +
            "\n\nRepair Instruction: Examine the traceback above. Fix the root cause in the affected file(s). "
            "Never delete or alter a test simply to bypass failure."
            f"{history_note}"
            f"{escalation_note}"
        )

        return {
            "verify_passed": False,
            "verify_attempts": attempts,
            "escalation_payload": escalation_payload,
            "diagnostic_report": diagnostic_report,
        }

    def _node_pass(self, state: "VerificationGraphState") -> "VerificationGraphState":
        """Graph node: terminal success — sync internal counters and return state unchanged."""
        self._last_passed = True
        self._attempt_count = 0
        return {"verify_passed": True, "verify_attempts": 0}

    def _node_hitl_escalation(self, state: "VerificationGraphState") -> "VerificationGraphState":
        """Graph node: raise a LangGraph ``interrupt()`` HITL checkpoint.

        Execution pauses here and the orchestrator receives the ``escalation_payload``
        dict.  On ``resume(approved=True)`` the graph continues; on ``approved=False``
        it terminates.  If no interrupt function is wired (e.g. in tests), the node
        logs the escalation and returns normally so the graph can still terminate.
        """
        payload: dict[str, Any] = state.get("escalation_payload") or {
            "action": "verification_escalation",
            "message": "Max verification attempts reached. Human review required.",
        }

        interrupt_fn = self._interrupt_fn or (interrupt if _LANGGRAPH_AVAILABLE else None)
        if interrupt_fn is not None:
            interrupt_fn(payload)

        # If interrupt_fn did not raise (e.g. in unit tests), fall through gracefully.
        return {"escalation_payload": payload}

    # ------------------------------------------------------------------
    # Conditional edge router
    # ------------------------------------------------------------------

    def _route_after_aggregate(self, state: "VerificationGraphState") -> str:
        """Conditional edge: route to ``pass_node`` on success, or to escalation/END on failure.

        Routing logic
        -------------
        - ``verify_passed = True``          → ``"pass_node"``
        - ``verify_attempts >= max_attempts`` → ``"hitl_escalation_node"``
        - otherwise                          → ``END``  (caller retries the whole graph)
        """
        if state.get("verify_passed"):
            return "pass_node"
        if (state.get("verify_attempts") or 0) >= self._max_attempts:
            return "hitl_escalation_node"
        return END

    # ------------------------------------------------------------------
    # Graph factory
    # ------------------------------------------------------------------

    def build_graph(self) -> Any:
        """Compile and return the verification ``StateGraph``.

        The returned object is a compiled LangGraph ``CompiledGraph`` that can be:
        - Invoked directly:  ``graph.invoke({"tier": "all"})``
        - Embedded as a node in the parent ``CodingAgentHarness`` graph via
          ``parent_graph.add_node("verify", graph)``

        Each node is registered with a ``RetryPolicy`` that retries only on
        transient infrastructure errors (timeout / missing binary).

        Raises
        ------
        RuntimeError
            If ``langgraph`` is not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "langgraph is not installed. Install it with: pip install langgraph"
            )

        # Shared RetryPolicy — retries only on transient OS-level errors
        _retry = RetryPolicy(
            max_attempts=3,
            initial_interval=1.0,
            backoff_factor=2.0,
            retry_on=_is_transient_error,
        )

        graph: StateGraph = StateGraph(VerificationGraphState)

        # ── Register nodes ────────────────────────────────────────────────
        graph.add_node("lint_node",           self._node_lint,           retry=_retry)
        graph.add_node("test_node",           self._node_test,           retry=_retry)
        graph.add_node("aggregate_node",      self._node_aggregate)
        graph.add_node("pass_node",           self._node_pass)
        graph.add_node("hitl_escalation_node", self._node_hitl_escalation)

        # ── Register edges ────────────────────────────────────────────────
        # Linear path: lint → test → aggregate
        graph.add_edge(START,          "lint_node")
        graph.add_edge("lint_node",    "test_node")
        graph.add_edge("test_node",    "aggregate_node")

        # Conditional branch after aggregation
        graph.add_conditional_edges(
            "aggregate_node",
            self._route_after_aggregate,
            {
                "pass_node":            "pass_node",
                "hitl_escalation_node": "hitl_escalation_node",
                END:                    END,
            },
        )

        # Terminal edges
        graph.add_edge("pass_node",            END)
        graph.add_edge("hitl_escalation_node", END)

        return graph.compile()


# ---------------------------------------------------------------------------
# Module-level convenience factories
# ---------------------------------------------------------------------------

def create_verification_tool(
    root_dir: str = ".",
    failures_path: str | None = None,
    interrupt_fn: Callable[[Any], Any] | None = None,
) -> StructuredTool:
    """Convenience factory returning the run_verification StructuredTool.

    Args:
        root_dir:     Workspace root used to locate harness/failures.md.
        failures_path: Override for the failures log path.
        interrupt_fn: Optional LangGraph ``interrupt`` callable. When provided,
                      a real HITL checkpoint is raised after ``max_attempts``
                      consecutive failures instead of only printing a warning.
    """
    toolset = VerificationToolset(
        root_dir=root_dir,
        failures_path=failures_path,
        interrupt_fn=interrupt_fn,
    )
    return toolset.get_tools()[0]


def build_verification_graph(
    root_dir: str = ".",
    max_attempts: int = 3,
    failures_path: str | None = None,
    interrupt_fn: Callable[[Any], Any] | None = None,
) -> Any:
    """Convenience factory that compiles and returns the verification StateGraph.

    Usage::

        graph = build_verification_graph(root_dir=".", max_attempts=3)

        # Run lint + tests
        result = graph.invoke({"tier": "all"})
        print(result["verify_passed"])     # True / False
        print(result["diagnostic_report"]) # Human-readable report

        # Embed in parent CodingAgentHarness graph
        parent_graph.add_node("verify", graph)

    Args:
        root_dir:     Workspace root — used for subprocess ``cwd`` and failure log path.
        max_attempts: Consecutive failure cap before HITL escalation (default: 3).
        failures_path: Override for harness/failures.md path.
        interrupt_fn: Optional LangGraph ``interrupt`` callable for HITL escalation.

    Returns:
        Compiled ``CompiledGraph[VerificationGraphState]``

    Raises:
        RuntimeError: If ``langgraph`` is not installed.
    """
    toolset = VerificationToolset(
        root_dir=root_dir,
        max_attempts=max_attempts,
        failures_path=failures_path,
        interrupt_fn=interrupt_fn,
    )
    return toolset.build_graph()
