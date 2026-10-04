"""
agent/core.py - CodingAgentHarness: single entry point for running and resuming agent executions.
"""

import logging
from typing import Any, AsyncIterator
from langgraph.types import Command
from frameworks.deepagents_backend import DeepAgentsBackend
from frameworks.langgraph_runtime import LangGraphRuntime
from frameworks.external_tools import ExternalToolsManager
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset
from tools.api_tools import APIToolset
from graphrag.ingest_tool import create_repo_ingest_tool
from graphrag.tool import create_graphrag_retriever_tool
from agent.guardrail import HarnessGuard
from agent.state import AgentState, create_initial_state
from agent.session_manager import SessionManager, SessionMetadata
from nodes.planning_subagent import planning_subagent
from nodes.design_subagent import design_subagent
from nodes.build_subagent import build_dev_subagent
from frameworks.evaluation import LangSmithTracer, HarnessEvaluator
from frameworks.agents_md_writer import generate_agents_md
from skills.registry import skill_library

logger = logging.getLogger(__name__)
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


# ---------------------------------------------------------------------------
# Supported LLM Models registry
# Includes cloud Ollama models and Groq models
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "groq:qwen/qwen3.8-27b": {
        "id": "groq:qwen/qwen3.8-27b",
        "provider": "groq",
        "name": "qwen/qwen3.8-27b",
        "label": "Qwen 3.8 27B (Groq)",
        "fallback": "ollama:gpt-oss:120b-cloud",
    },
    "ollama:gemma4:cloud": {
        "id": "ollama:gemma4:cloud",
        "provider": "ollama",
        "name": "gemma4:cloud",
        "label": "Gemma 4 (Cloud)",
        "aliases": ["gemma4:cloud", "gemma4", "ollama:gemma4"],
        "fallback": "groq:qwen/qwen3.8-27b",
    },
    "ollama:nvidia-nemotron-super:cloud": {
        "id": "ollama:nvidia-nemotron-super:cloud",
        "provider": "ollama",
        "name": "nvidia-nemotron-super:cloud",
        "label": "NVIDIA Nemotron Super (Cloud)",
        "aliases": [
            "nvidia-nemotron-super:cloud",
            "nemotron-super:cloud",
            "nemotron-3-super:cloud",
            "ollama:nemotron-3-super:cloud",
        ],
        "fallback": "groq:qwen/qwen3.8-27b",
    },
    "ollama:gpt-oss:120b-cloud": {
        "id": "ollama:gpt-oss:120b-cloud",
        "provider": "ollama",
        "name": "gpt-oss:120b-cloud",
        "label": "GPT-OSS 120B (Cloud)",
        "aliases": [
            "gpt-oss-120b:cloud",
            "gpt-oss:120b-cloud",
            "ollama:gpt-oss-120b:cloud",
        ],
        "fallback": "groq:qwen/qwen3.8-27b",
    },
}


def normalize_model_identifier(model_name: str | None) -> str:
    """Normalize user or UI supplied model identifier to registered model key.

    Maps aliases such as 'gemma4:cloud', 'nvidia-nemotron-super:cloud',
    'gpt-oss-120b:cloud', etc., to their canonical representation.
    """
    if not model_name:
        return "groq:qwen/qwen3.8-27b"

    cleaned = model_name.strip()
    if cleaned in SUPPORTED_MODELS:
        return cleaned

    # Check aliases
    for canonical_id, config in SUPPORTED_MODELS.items():
        aliases = config.get("aliases", [])
        if cleaned in aliases or cleaned.lower() in [a.lower() for a in aliases]:
            return canonical_id

    # If it starts with provider prefix or is unrecognized, retain as-is or auto-prefix
    if ":" in cleaned:
        return cleaned
    return f"ollama:{cleaned}"


class CodingAgentHarness:
    """The central harness orchestrator.

    Owns the backend, subagents, and checkpointer.
    Provides run() and resume() thread-scoped entry points.
    """

    SUPPORTED_MODELS = SUPPORTED_MODELS

    def __init__(
        self,
        root_dir: str = ".",
        model: str = "groq:qwen/qwen3.8-27b",
        tools: list[Any] | None = None,
        tracer: Any = None,
        sandbox_mode: str | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._model = normalize_model_identifier(model)
        self._tracer = tracer or LangSmithTracer()
        self._evaluator = HarnessEvaluator(tracer_instance=self._tracer)

        self._guard = HarnessGuard(f"{root_dir}/harness/permissions.json", silent_mode=True)
        self._backend = DeepAgentsBackend(
            root_dir=root_dir,
            guard=self._guard,
            sandbox_mode=sandbox_mode,
        )
        self._sessions = SessionManager(root_dir=root_dir)
        self._runtime = LangGraphRuntime(session_manager=self._sessions, root_dir=root_dir)
        self._external_tools = ExternalToolsManager()
        self._deploy_tools = DeployToolset(root_dir=root_dir)
        self._verification_tools = VerificationToolset(root_dir=root_dir)
        self._api_tools = APIToolset()
        # Build tools list with safety wraps for DB-dependent tools
        try:
            self._tools = (
                tools
                if tools is not None
                else [
                    *self._deploy_tools.get_tools(),
                    *self._verification_tools.get_tools(),
                    *self._api_tools.get_tools(),
                ]
            )
            # Add DB-dependent tools separately to prevent boot-loops
            try:
                self._tools.append(create_repo_ingest_tool())
                self._tools.append(create_graphrag_retriever_tool())
            except Exception as db_exc:
                logger.error(f"GraphRAG tools failed to load: {db_exc}")
        except Exception as e:
            logger.error(f"Base tools initialization failed: {e}")
            self._tools = []

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
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def sessions(self) -> SessionManager:
        return self._sessions

    @property
    def runtime(self) -> LangGraphRuntime:
        return self._runtime

    @property
    def tracer(self) -> LangSmithTracer:
        return self._tracer

    @property
    def evaluator(self) -> HarnessEvaluator:
        return self._evaluator

    @property
    def model(self) -> str:
        """Return the default active model identifier."""
        return self._model

    @property
    def supported_models(self) -> dict[str, dict[str, Any]]:
        """Return dictionary of supported models."""
        return self.SUPPORTED_MODELS

    @property
    def skills(self):
        """Access the global SkillLibrary to register or load custom skills.

        Example::

            from skills.schema import Skill
            harness.skills.register(Skill(
                name="my-convention",
                description="Team coding conventions",
                tags=["python", "style"],
                instructions="Always use Black. Max line 88. Type-annotate all functions.",
            ))
            harness.skills.load_from_directory("./my_skills/")
        """
        return skill_library

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

    async def run(self, user_request: str, thread_id: str, model: str | None = None) -> dict[str, Any]:
        """Executes a run for a specific thread with an optional model override.

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
            input_payload: Any = {"messages": [{"role": "user", "content": user_request}]}
        else:
            # First turn: initialize full AgentState
            input_payload = create_initial_state(user_request)

        self._sessions.update_session(thread_id, increment_messages=True, status="active")
        await self._runtime.persist_session_to_memory(thread_id)

        # ── Pre-flight: auto-generate a query-specific AGENTS.md ──────────────
        # Runs before the agent graph so all subagents (planning, design, build)
        # receive directives tailored to this exact user request.
        chosen_model = normalize_model_identifier(model) if model else self._model
        await generate_agents_md(
            user_request=user_request,
            root_dir=self._root_dir,
            model=chosen_model,
        )

        agent_to_invoke = self._agent
        if chosen_model != self._model:
            # Dynamically build agent with requested model override
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
            agent_to_invoke = self._backend.build_agent(
                model=chosen_model,
                subagents=subagents,
                system_prompt=(
                    "You are an autonomous coding harness agent. Always follow AGENTS.md, "
                    "respect permissions.json, coordinate planning -> design -> build, and verify all code."
                ),
                checkpointer=self._runtime.checkpointer,
                state_schema=AgentState,
            )

        result = await agent_to_invoke.ainvoke(
            input_payload,
            config=config,
        )

        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        self._sessions.update_session(thread_id, status="active")
        await self._runtime.persist_session_to_memory(thread_id)
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
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        self._sessions.update_session(thread_id, status="active")
        await self._runtime.persist_session_to_memory(thread_id)
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
