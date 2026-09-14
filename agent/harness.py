"""
agent/harness.py - Guardrails enforcement, HITL interrupt coordination, and failure logging.

HarnessGuard implements a three-gate pipeline for every agent action:

    Gate 1 — DENY  : action matches a deny pattern → immediately blocked, logged to failures.md
    Gate 2 — ASK   : action matches an ask pattern  → human approval required via LangGraph
                     interrupt(); rejection is logged to failures.md
    Gate 3 — ALLOW : action matches an allow pattern → executed without interruption

All rules are pattern-matched (fnmatch glob) against the action signature string, e.g.::

    Bash(git push origin main)
    mcp__github__create_pull_request
    vercel_deploy

Rules are loaded from ``harness/permissions.json`` and hot-reloaded when the file changes.
Every denied or rejected event is appended to ``harness/failures.md`` as a structured row.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Failure log helpers
# ---------------------------------------------------------------------------

_FAILURES_HEADER = (
    "# Failure Log and Triage\n\n"
    "| Timestamp | Stage | Category | Description | Resolution / Action Taken |\n"
    "|---|---|---|---|---|\n"
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _append_failure(
    failures_path: Path,
    stage: str,
    category: str,
    description: str,
    resolution: str = "—",
) -> None:
    """Append one row to failures.md, creating/re-initialising the file if needed."""
    if not failures_path.exists() or failures_path.stat().st_size == 0:
        failures_path.write_text(_FAILURES_HEADER, encoding="utf-8")

    row = (
        f"| {_now_utc()} "
        f"| {stage} "
        f"| {category} "
        f"| {description} "
        f"| {resolution} |\n"
    )
    with failures_path.open("a", encoding="utf-8") as fh:
        fh.write(row)


# ---------------------------------------------------------------------------
# HarnessGuard
# ---------------------------------------------------------------------------

class HarnessGuard:
    """Loads permissions.json and enforces allow / ask / deny rules on agent actions.

    Usage (inside a LangGraph node / tool)::

        guard = HarnessGuard("harness/permissions.json")

        # Simple permission check (no side-effects):
        level = guard.check_permission("Bash(git push origin main)")  # -> "ask"

        # Full HITL enforcement — raises on deny/rejection, passes on allow/approval:
        from langgraph.types import interrupt
        guard.enforce("Bash(git push origin main)", interrupt_fn=interrupt)
    """

    def __init__(
        self,
        permissions_path: str = "harness/permissions.json",
        failures_path: str | None = None,
    ) -> None:
        self._path = Path(permissions_path)
        self._failures_path = Path(
            failures_path
            if failures_path is not None
            else self._path.parent / "failures.md"
        )
        self._permissions: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
        self._mtime: float = 0.0
        self.load()

    # ------------------------------------------------------------------
    # Loading & hot-reload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read permissions.json from disk. Silently no-ops if the file is missing."""
        if not self._path.exists():
            return
        try:
            mtime = self._path.stat().st_mtime
            data = json.loads(self._path.read_text(encoding="utf-8"))

            # Schema guard: the file must be a dict whose values are lists of strings.
            # A malformed file (e.g. "allow": "Bash(*)" instead of ["Bash(*)"])
            # would silently iterate over characters and match nothing, effectively
            # failing open.  Reject anything that doesn't pass this check.
            if not isinstance(data, dict):
                raise ValueError("permissions.json must be a JSON object at the top level.")
            validated: dict[str, list[str]] = {"allow": [], "ask": [], "deny": []}
            for key in ("allow", "ask", "deny"):
                raw = data.get(key, [])
                if not isinstance(raw, list):
                    raise ValueError(
                        f"permissions.json: '{key}' must be a list, got {type(raw).__name__}."
                    )
                if not all(isinstance(p, str) for p in raw):
                    raise ValueError(
                        f"permissions.json: all entries in '{key}' must be strings."
                    )
                validated[key] = raw

            self._permissions = validated
            self._mtime = mtime
        except Exception:
            pass  # Keep previous rules if the file is temporarily malformed or mid-write

    def _reload_if_changed(self) -> None:
        """Hot-reload permissions.json if the file has been modified since last load."""
        if not self._path.exists():
            return
        try:
            if self._path.stat().st_mtime > self._mtime:
                self.load()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Core permission check (read-only, no side-effects)
    # ------------------------------------------------------------------

    def check_permission(self, action_signature: str) -> str:
        """Return 'deny', 'ask', or 'allow' for the given action signature.

        Rules are evaluated in priority order: deny > ask > allow.
        If no pattern matches, defaults to 'ask' (safe / HITL required).

        Pattern matching uses Unix shell-style wildcards (fnmatch)::

            Bash(git push*)   matches  Bash(git push origin main)
            mcp__github__*    matches  mcp__github__create_pull_request

        The signature is whitespace-normalised before matching so that padded
        or multi-space variants like ``Bash(   rm  -rf  /)`` cannot bypass a
        deny rule written as ``Bash(*rm -rf*)``.  Internal whitespace is
        collapsed to a single space; leading/trailing space is stripped.

        ``permissions.json`` is read from disk on **every call** (via ``load()``)
        so that rule changes are always reflected immediately — no mtime cache,
        no clock-skew window, no stale-rule risk between two writes on the
        same filesystem timestamp.
        """
        self.load()  # Always read fresh from disk — security over cache speed

        # Normalise: collapse internal whitespace so crafted padding can't
        # slip past a deny pattern (e.g. "Bash(  rm  -rf /)" → "Bash(rm -rf /)").
        action_signature = " ".join(action_signature.split())

        for pattern in self._permissions.get("deny", []):
            if fnmatch(action_signature, pattern):
                return "deny"

        for pattern in self._permissions.get("ask", []):
            if fnmatch(action_signature, pattern):
                return "ask"

        for pattern in self._permissions.get("allow", []):
            if fnmatch(action_signature, pattern):
                return "allow"

        return "ask"  # Default: require human approval for unrecognised actions

    # ------------------------------------------------------------------
    # HITL enforcement pipeline (deny → ask → allow)
    # ------------------------------------------------------------------

    def enforce(
        self,
        action_signature: str,
        interrupt_fn: Callable[[Any], Any] | None = None,
        *,
        description: str | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Run the full three-gate HITL pipeline for an action.

        Args:
            action_signature: The canonical string identifying the action,
                              e.g. ``"Bash(git push origin main)"``.
            interrupt_fn:     The LangGraph ``interrupt`` callable (or any
                              callable that blocks until the human responds).
                              If omitted (None), defaults to ``langgraph.types.interrupt``.
            description:      Human-readable explanation shown in the approval
                              prompt. Defaults to the signature.
            extra_context:    Additional key-value pairs included in the
                              interrupt payload for richer UI display.

        Raises:
            PermissionError:  Gate 1 — action is in the deny list.
            PermissionError:  Gate 2 — action required approval and was rejected.
            ValueError:       Gate 2 — approval required but no interrupt_fn provided and LangGraph unavailable.

        Returns:
            None on success (action is approved / unconditionally allowed).
        """
        permission = self.check_permission(action_signature)
        label = description or action_signature

        # ── Gate 1: Hard deny ────────────────────────────────────────────
        if permission == "deny":
            msg = (
                f"🛑 [SECURITY DENIED] '{action_signature}' is blocked by harness "
                f"guardrails (deny rule in permissions.json). Execution aborted."
            )
            
            _append_failure(
                self._failures_path,
                stage="guardrail",
                category="security-deny",
                description=label,
                resolution="Blocked by deny rule — no execution.",
            )
            raise PermissionError(msg)

        # ── Gate 2: Human-in-the-Loop ────────────────────────────────────
        if permission == "ask":
            fn = interrupt_fn
            if fn is None:
                try:
                    from langgraph.types import interrupt as _lg_interrupt
                    fn = _lg_interrupt
                except ImportError:
                    raise ValueError(
                        "Action requires human approval ('ask'), but no interrupt_fn was provided "
                        "and langgraph.types.interrupt could not be imported."
                    )

            payload: dict[str, Any] = {
                "action": "hitl_approval_required",
                "signature": action_signature,
                "description": label,
                "permission_level": "ask",
            }
            if extra_context:
                payload.update(extra_context)

            approval = fn(payload)

            # Normalise various response shapes: bool, dict, str
            is_approved: bool = False
            if isinstance(approval, bool):
                is_approved = approval
            elif isinstance(approval, dict):
                is_approved = bool(approval.get("approved", False))
            elif isinstance(approval, str):
                is_approved = approval.strip().lower() in {"yes", "true", "approve", "approved"}

            if not is_approved:
                msg = (
                    f"⚠️ [HITL REJECTED] '{action_signature}' was rejected by the human "
                    f"operator. Execution aborted."
                )
                _append_failure(
                    self._failures_path,
                    stage="hitl",
                    category="human-rejected",
                    description=label,
                    resolution="Rejected by operator via HITL interrupt.",
                )
                raise PermissionError(msg)

        # ── Gate 3: Allow (or approved) ──────────────────────────────────
        # Nothing to do — caller proceeds with execution.

    # ------------------------------------------------------------------
    # Convenience: log an arbitrary failure (e.g. from verification_tools)
    # ------------------------------------------------------------------

    def log_failure(
        self,
        stage: str,
        category: str,
        description: str,
        resolution: str = "—",
    ) -> None:
        """Append an arbitrary failure row to failures.md.

        Useful for verification_tools, deploy_tools, or any harness component
        that needs structured failure tracking without running through the
        permission pipeline.

        Example::

            guard.log_failure(
                stage="verification",
                category="test-failure",
                description="pytest returned exit code 1",
                resolution="Traceback written to stdout; agent retrying fix.",
            )
        """
        _append_failure(
            self._failures_path,
            stage=stage,
            category=category,
            description=description,
            resolution=resolution,
        )

    # ------------------------------------------------------------------
    # Tool wrapping — apply guard to any LangChain StructuredTool
    # ------------------------------------------------------------------

    def wrap_tool_with_guard(
        self,
        tool: Any,
        interrupt_fn: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Wrap a LangChain StructuredTool so guard.enforce() fires before its func.

        The action signature used for permission matching is ``tool.name``, e.g.
        ``mcp__github__create_pull_request`` — matching patterns like
        ``"mcp__github__*"`` already present in permissions.json.

        If the tool does not expose a ``.func`` attribute (e.g. it is an
        :class:`langchain_core.tools.BaseTool` subclass with a custom
        ``_run`` method) the original tool is returned unchanged so the caller
        can still use it — just without the enforce gate.

        Args:
            tool:         Any LangChain BaseTool / StructuredTool instance.
            interrupt_fn: The LangGraph ``interrupt`` callable (or any callable
                          that blocks until the human responds). If None, defaults
                          to ``langgraph.types.interrupt``.

        Returns:
            A new StructuredTool with an identical schema and description but
            with a guarded wrapper around the original func.
        """
        import functools

        try:
            from langchain_core.tools import StructuredTool as _ST
        except ImportError:
            return tool  # langchain_core not installed — pass through

        original_func: Callable[..., Any] | None = getattr(tool, "func", None)
        if original_func is None:
            return tool  # Not a StructuredTool with .func — return as-is

        tool_name: str = getattr(tool, "name", "unknown_tool")
        tool_desc: str = getattr(tool, "description", "")
        tool_schema: Any = getattr(tool, "args_schema", None)

        guard_self = self  # capture for closure

        @functools.wraps(original_func)
        def _guarded(*args: Any, **kwargs: Any) -> Any:
            guard_self.enforce(tool_name, interrupt_fn=interrupt_fn)
            return original_func(*args, **kwargs)

        wrapped_kwargs: dict[str, Any] = {
            "func": _guarded,
            "name": tool_name,
            "description": f"[HITL-guarded] {tool_desc}",
        }
        if tool_schema is not None:
            wrapped_kwargs["args_schema"] = tool_schema

        return _ST.from_function(**wrapped_kwargs)
