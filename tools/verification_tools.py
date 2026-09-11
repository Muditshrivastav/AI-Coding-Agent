"""
tools/verification_tools.py - Self-verification toolset for code validation and repair loops.
Executes linting, static checks, and unit tests, capturing structured failure tracebacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import shutil
import subprocess
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class VerificationArgs(BaseModel):
    tier: str = Field(
        default="all",
        description="Verification tier to run: 'all' (lint + test), 'lint' (ruff static checks), or 'test' (pytest suite).",
    )
    test_target: str | None = Field(
        default=None,
        description="Optional specific test file or path to run (e.g. 'tests/test_build_subagent.py').",
    )


class VerificationToolset:
    """Manages programmatic self-verification execution, failure logging to harness/failures.md, and error feedback."""

    def __init__(
        self,
        root_dir: str = ".",
        max_attempts: int = 3,
        failures_path: str | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._max_attempts = max_attempts
        self._failures_path = failures_path or os.path.join(root_dir, "harness", "failures.md")
        self._attempt_count: int = 0
        self._last_passed: bool = False

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

    def _run_subprocess(self, cmd: list[str]) -> tuple[int, str, str]:
        """Runs a command inside root_dir, returning (returncode, stdout, stderr)."""
        try:
            res = subprocess.run(
                cmd,
                cwd=self._root_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Execution timed out after 120 seconds."
        except FileNotFoundError:
            return 127, "", f"Command '{cmd[0]}' not found."
        except Exception as e:
            return 1, "", str(e)

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

    def run_verification(self, tier: str = "all", test_target: str | None = None) -> str:
        """Executes verification checks and formats actionable diagnostics for repair loops."""
        self._attempt_count += 1
        tier = (tier or "all").lower()

        results: list[dict[str, Any]] = []
        has_make = shutil.which("make") is not None

        # Execute based on tier
        if tier in ("all", "lint"):
            if has_make and tier == "all" and not test_target:
                rc, out, err = self._run_subprocess(["make", "verify"])
                results.append({"step": "make verify", "returncode": rc, "stdout": out, "stderr": err})
            else:
                rc_check, out_check, err_check = self._run_subprocess(["ruff", "check", "."])
                if rc_check == 127:
                    rc_check, out_check, err_check = self._run_subprocess(["python", "-m", "ruff", "check", "."])
                results.append({"step": "ruff check", "returncode": rc_check, "stdout": out_check, "stderr": err_check})

        if tier in ("all", "test"):
            # Skip if make verify already ran both and succeeded
            if not (has_make and tier == "all" and not test_target and results and results[0]["returncode"] == 0):
                test_cmd = ["pytest", "-q"]
                if test_target:
                    test_cmd.append(test_target)
                rc_test, out_test, err_test = self._run_subprocess(test_cmd)
                if rc_test == 127:
                    test_cmd[0:1] = ["python", "-m", "pytest"]
                    rc_test, out_test, err_test = self._run_subprocess(test_cmd)
                results.append({"step": "pytest", "returncode": rc_test, "stdout": out_test, "stderr": err_test})

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


def create_verification_tool(root_dir: str = ".", failures_path: str | None = None) -> StructuredTool:
    """Convenience factory returning the run_verification tool."""
    toolset = VerificationToolset(root_dir=root_dir, failures_path=failures_path)
    return toolset.get_tools()[0]

