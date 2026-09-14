"""
agent/core.py - CodingAgentHarness: single entry point for running and resuming agent executions.
"""

from typing import Any, AsyncIterator
from langgraph.types import Command
from frameworks.deepagents_backend import DeepAgentsBackend
from frameworks.langgraph_runtime import LangGraphRuntime
from frameworks.external_tools import ExternalToolsManager
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset
from agent.guardrail import HarnessGuard
from agent.state import AgentState, create_initial_state
from agent.session_manager import SessionManager, SessionMetadata
from nodes.planning_subagent import planning_subagent
from nodes.design_subagent import design_subagent
from nodes.build_subagent import build_dev_subagent

# ---------------------------------------------------------------------------
# Input sanitization — known prompt-injection trigger phrases.
# These are heuristic patterns, not a complete defence; they serve as a
# first-layer trip-wire that logs and blocks the most obvious attacks.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous",
    "ignore all previous",
    "ignore your instructions",
    "forget your instructions",
    "disregard your instructions",
    "new instructions:",
    "system:",
    "<|im_sep|>",
    "<|endoftext|>",
    "<|system|>",
    "act as if you",
    "you are now",
    "jailbreak",
)


class CodingAgentHarness:
    """The central harness orchestrator.

    Owns the backend, subagents, and checkpointer.
    Provides run() and resume() thread-scoped entry points.
    """

    def __init__(
        self,
        root_dir: str = ".",
        model: str = "anthropic:claude-sonnet-4-6",
        tools: list[Any] | None = None,
        tracer: Any = None,
    ) -> None:
        self._root_dir = root_dir
        self._model = model
        self._tracer = tracer

        self._guard = HarnessGuard(f"{root_dir}/harness/permissions.json")
        self._backend = DeepAgentsBackend(root_dir=root_dir, guard=self._guard)
        self._runtime = LangGraphRuntime()
        self._sessions = SessionManager(root_dir=root_dir)
        self._external_tools = ExternalToolsManager()
        self._deploy_tools = DeployToolset(root_dir=root_dir)
        self._verification_tools = VerificationToolset(root_dir=root_dir)
        self._tools = (
            tools
            if tools is not None
            else [*self._deploy_tools.get_tools(), *self._verification_tools.get_tools()]
        )

        # Build subagents hierarchy
        subagents = [
            planning_subagent,
            design_subagent,
            build_dev_subagent(
                self._tools,
                external_tools=self._external_tools,
                root_dir=self._root_dir,
                backend=self._backend.backend,
                guard=self._guard,
            ),
        ]

        self._agent = self._backend.build_agent(
            model=self._model,
            subagents=subagents,
            system_prompt=(
                "You are an autonomous coding harness agent. Always follow AGENTS.md, "
                "respect permissions.json, coordinate planning -> design -> build, and verify all code."
            ),
            checkpointer=self._runtime.checkpointer,
            state_schema=AgentState,
        )

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def guard(self) -> HarnessGuard:
        return self._guard

    @property
    def sessions(self) -> SessionManager:
        return self._sessions

    # ------------------------------------------------------------------
    # Input sanitization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_suspicious_input(text: str) -> str | None:
        """Return the matched injection pattern if *text* looks suspicious, else None.

        Checks are case-insensitive and match substrings so partial injection
        attempts (e.g. padded with whitespace) are still caught.
        """
        lower = text.lower()
        for pattern in _INJECTION_PATTERNS:
            if pattern in lower:
                return pattern
        return None

    async def run(self, user_request: str, thread_id: str) -> dict[str, Any]:
        """Executes a run for a specific thread.

        Performs a prompt-injection pre-flight check before handing the request
        to the agent. Suspicious inputs are logged to harness/failures.md and
        returned immediately as ``{"status": "blocked"}`` without reaching the
        agent graph.
        """
        # ── Pre-flight: prompt-injection guard ─────────────────────────────────
        matched = self._is_suspicious_input(user_request)
        if matched:
            preview = user_request[:120].replace("\n", " ")
            self._guard.log_failure(
                stage="input",
                category="prompt-injection",
                description=f"Matched pattern '{matched}' in: {preview}",
                resolution="Request blocked before reaching agent.",
            )
            return {
                "status": "blocked",
                "reason": (
                    f"Input flagged as potential prompt injection "
                    f"(matched pattern: '{matched}'). Request not forwarded to agent."
                ),
            }

        config = {"configurable": {"thread_id": thread_id}}

        # Track or register the session in SessionManager
        if not self._sessions.has_session(thread_id):
            # Derive title from first request preview
            title_preview = user_request.strip().split("\n")[0][:40]
            self._sessions.create_session(title=title_preview or f"Session {thread_id[:8]}", session_id=thread_id)

        # Check if thread has prior state in the checkpointer
        existing_state = await self.get_state(thread_id)
        if existing_state and "messages" in existing_state:
            # Subsequent turn in this session: append user message to existing history
            input_payload: dict[str, Any] = {"messages": [{"role": "user", "content": user_request}]}
        else:
            # First turn: initialize full AgentState
            input_payload = create_initial_state(user_request)

        self._sessions.update_session(thread_id, increment_messages=True, status="active")

        result = await self._agent.ainvoke(
            input_payload,
            config=config,
        )

        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        self._sessions.update_session(thread_id, status="active")
        return {"status": "complete", "result": result}

    async def resume(self, thread_id: str, approved: bool) -> dict[str, Any]:
        """Resumes an interrupted execution upon HITL decision."""
        config = {"configurable": {"thread_id": thread_id}}
        result = await self._agent.ainvoke(
            Command(resume={"approved": approved}),
            config=config,
        )
        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        self._sessions.update_session(thread_id, status="active")
        return {"status": "complete", "result": result}

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Retrieves the current state of a thread from checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await self._agent.aget_state(config)
            return snapshot.values if snapshot else None
        except Exception:
            return None

    async def stream(self, thread_id: str) -> AsyncIterator[Any]:
        """Streams updates for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self._agent.astream(None, config=config):
            yield chunk
