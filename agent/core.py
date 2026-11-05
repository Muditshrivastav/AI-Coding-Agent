"""
agent/core.py - CodingAgentHarness: single entry point for running and resuming agent executions.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command, interrupt

from agent.guardrail import HarnessGuard
from agent.repair_loop import build_repair_graph
from agent.session_manager import SessionManager
from agent.state import AgentState
from frameworks.agents_md_writer import generate_agents_md
from frameworks.deepagents_backend import DeepAgentsBackend
from frameworks.evaluation import HarnessEvaluator, LangSmithTracer
from frameworks.external_tools import ExternalToolsManager
from frameworks.langgraph_runtime import LangGraphRuntime
from frameworks.model_config import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    GROQ_DEFAULT_MODEL,
)
from graphrag.ingest_tool import create_repo_ingest_tool
from graphrag.tool import create_graphrag_retriever_tool
from nodes.build_subagent import build_dev_subagent, get_build_dev_tools
from nodes.design_subagent import design_subagent, make_design_subagent
from nodes.planning_subagent import make_planning_subagent, planning_subagent
from skills.registry import skill_library
from tools.api_tools import APIToolset
from tools.deploy_tools import DeployToolset
from tools.verification_tools import VerificationToolset

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Input sanitization — known prompt-injection trigger phrases.
# These are heuristic patterns, not a complete defence; they serve as a
# first-layer trip-wire that logs and blocks the most obvious attacks.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore previous",
        re.compile(
            r"\bignore\s+(?:all\s+)?previous\s+(?:system\s+)?instructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ignore your instructions",
        re.compile(r"\bignore\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "forget your instructions",
        re.compile(r"\bforget\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "disregard your instructions",
        re.compile(r"\bdisregard\s+your\s+instructions?\b", re.IGNORECASE),
    ),
    (
        "new instructions:",
        re.compile(
            r"(?im)^\s*new instructions\s*:\s*(?:ignore|disregard|override|reveal)\b"
        ),
    ),
    (
        "system:",
        re.compile(
            r"(?im)^\s*system\s*:\s*(?:ignore|disregard|override|you are now|reveal)\b"
        ),
    ),
    ("<|im_sep|>", re.compile(r"<\|im_sep\|>", re.IGNORECASE)),
    ("<|endoftext|>", re.compile(r"<\|endoftext\|>", re.IGNORECASE)),
    ("<|system|>", re.compile(r"<\|system\|>", re.IGNORECASE)),
    (
        "act as if you",
        re.compile(r"\bact\s+as\s+if\s+you\s+(?:are|were|can|must)\b", re.IGNORECASE),
    ),
    (
        "you are now",
        re.compile(
            r"(?im)^\s*you\s+are\s+now\s+(?:an?\s+)?(?:assistant|system|agent)\b"
        ),
    ),
    (
        "jailbreak",
        re.compile(
            r"\b(?:enable|perform|attempt)\s+(?:a\s+)?jailbreak\b", re.IGNORECASE
        ),
    ),
)


# ---------------------------------------------------------------------------
# Supported LLM Models registry
# Includes cloud Ollama models and Groq models
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    GROQ_DEFAULT_MODEL: {
        "id": GROQ_DEFAULT_MODEL,
        "provider": "groq",
        "name": GROQ_DEFAULT_MODEL.removeprefix("groq:"),
        "label": "Qwen 3.8 27B (Groq)",
        "fallback": FALLBACK_MODEL,
    },
    "ollama:gemma4:cloud": {
        "id": "ollama:gemma4:cloud",
        "provider": "ollama",
        "name": "gemma4:cloud",
        "label": "Gemma 4 (Cloud)",
        "aliases": ["gemma4:cloud", "gemma4", "ollama:gemma4"],
        "fallback": GROQ_DEFAULT_MODEL,
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
        "fallback": GROQ_DEFAULT_MODEL,
    },
    FALLBACK_MODEL: {
        "id": FALLBACK_MODEL,
        "provider": "ollama",
        "name": FALLBACK_MODEL.removeprefix("ollama:"),
        "label": "GPT-OSS 120B (Cloud)",
        "aliases": [
            "gpt-oss-120b:cloud",
            FALLBACK_MODEL.removeprefix("ollama:"),
            FALLBACK_MODEL,
        ],
        "fallback": GROQ_DEFAULT_MODEL,
    },
}


def normalize_model_identifier(model_name: str | None) -> str:
    """Normalize user or UI supplied model identifier to registered model key.

    Maps aliases such as 'gemma4:cloud', 'nvidia-nemotron-super:cloud',
    'gpt-oss-120b:cloud', etc., to their canonical representation.
    """
    if not model_name:
        return DEFAULT_MODEL

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
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        tracer: Any = None,
        sandbox_mode: str | None = None,
        runtime: LangGraphRuntime | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._model = normalize_model_identifier(model)
        self._tracer = tracer or LangSmithTracer()
        self._evaluator = HarnessEvaluator(tracer_instance=self._tracer)

        self._guard = HarnessGuard(
            f"{root_dir}/harness/permissions.json",
            silent_mode=True,
            sandbox_mode=sandbox_mode,
        )
        self._backend = DeepAgentsBackend(
            root_dir=root_dir,
            guard=self._guard,
            sandbox_mode=sandbox_mode,
        )
        self._runtime = runtime or LangGraphRuntime(root_dir=root_dir)
        self._sessions = self._runtime.session_manager
        self._execution_graphs: dict[str, Any] = {}
        self._external_tools = ExternalToolsManager()
        self._mcp_clients: list[Any] = []
        self._drawio_client: Any = None
        self._planner_agent: Any = None
        self._design_agent: Any = None
        self._loaded_tool_names: list[str] = []
        self._additional_mcp_status: dict[str, str] = {}
        self._deploy_tools = DeployToolset(root_dir=root_dir)
        self._verification_tools = VerificationToolset(root_dir=root_dir)
        self._api_tools = APIToolset()
        self._system_prompt = (
            "You are an autonomous coding harness agent. Always follow AGENTS.md and TASK.md, "
            "respect permissions.json, coordinate planning -> design -> build, and verify all code.\n"
            "When the task requires a shell command or file mutation, invoke the appropriate tool; "
            "do not skip it or ask for approval in chat. HarnessGuard checks the action before it "
            "runs and will pause for the user through the approval UI when the rule is 'ask'. "
            "After approval, continue from the resumed tool result. If denied, do not retry through "
            "another tool or method.\n"
            "For Git, package-install, test, or build commands, use execute_command directly or "
            "delegate to the build-agent. Do not claim that shell or Git access is unavailable "
            "before attempting a tool; let HarnessGuard request approval when required.\n"
            "For code changes, call run_verification after editing. If checks fail, "
            "diagnose and repair the reported issue, then verify again.\n"
            "FILESYSTEM PATH RULES: All filesystem tools (ls, read_file, write_file, edit_file) require "
            "POSIX virtual paths starting with /. Never use Windows paths like C:\\\\path. "
            "The workspace root is /. Use ls(path='/') to explore, read_file(file_path='/agent/core.py'), etc."
        )
        # Build tools list with safety wraps for DB-dependent tools
        try:
            self._tools = (
                list(tools)
                if tools is not None
                else [
                    *self._deploy_tools.get_tools(),
                    *self._verification_tools.get_tools(),
                    *self._api_tools.get_tools(),
                ]
            )
            # B7: build a single shared GraphRAGPipeline so ingest and retrieval use the same
            # store (critical for in-memory fallback mode where they previously shared nothing)
            # and so we only pay the cost of one reranker + embedder load.
            try:
                from graphrag.pipeline import GraphRAGPipeline

                shared_pipeline = GraphRAGPipeline()
                self._tools.append(create_repo_ingest_tool(pipeline=shared_pipeline))
                self._tools.append(
                    create_graphrag_retriever_tool(pipeline=shared_pipeline)
                )
            except Exception as db_exc:  # noqa: BLE001 - optional GraphRAG integration is isolated and logged
                logger.error(f"GraphRAG tools failed to load: {db_exc}")
        except Exception as e:  # noqa: BLE001 - tool initialization failure is logged before degraded startup
            logger.error(f"Base tools initialization failed: {e}")
            self._tools = []

        self._subagents: list[dict[str, Any]] = []
        self._loaded_tool_names = sorted(
            {getattr(tool, "name", str(tool)) for tool in self._tools}
        )
        self._agent: Any = None

    @classmethod
    async def create(
        cls,
        root_dir: str = ".",
        model: str = DEFAULT_MODEL,
        tools: list[Any] | None = None,
        tracer: Any = None,
        sandbox_mode: str | None = None,
    ) -> CodingAgentHarness:
        """Create a harness and load optional MCP integrations before building agents."""
        sessions = SessionManager(root_dir=root_dir)
        runtime = await LangGraphRuntime.create(
            session_manager=sessions,
            root_dir=root_dir,
        )
        interrupted = sessions.mark_running_interrupted()
        if interrupted:
            logger.warning(
                "Recovered %d interrupted run(s) from session storage.",
                len(interrupted),
            )
        harness = cls(
            root_dir=root_dir,
            model=model,
            tools=tools,
            tracer=tracer,
            sandbox_mode=sandbox_mode,
            runtime=runtime,
        )
        try:
            (
                build_tools,
                external_tools,
                mcp_clients,
            ) = await get_build_dev_tools(
                base_tools=harness._tools,
                external_tools_manager=harness._external_tools,
                root_dir=root_dir,
                guard=harness._guard,
                backend=harness._backend.backend,
            )
            planner = make_planning_subagent(root_dir=root_dir, guard=harness._guard)
            designer, drawio_client = await make_design_subagent(
                root_dir=root_dir,
                guard=harness._guard,
            )
            harness._planner_agent = planner
            harness._design_agent = designer
            harness._drawio_client = drawio_client
            harness._additional_mcp_status = {
                "drawio": (
                    f"loaded {len(drawio_client.tool_names)} tools"
                    if drawio_client.tool_names
                    else "unavailable"
                )
            }
            harness._external_tools = external_tools
            harness._mcp_clients = mcp_clients
            harness._subagents = [
                {
                    "name": "planning-agent",
                    "description": planning_subagent["description"],
                    "runnable": planner,
                },
                {
                    "name": "design-agent",
                    "description": design_subagent["description"],
                    "runnable": designer,
                },
                build_dev_subagent(
                    build_tools,
                    external_tools=external_tools,
                    mcp_clients=mcp_clients,
                    root_dir=root_dir,
                    backend=harness._backend.backend,
                    guard=harness._guard,
                    tools_already_guarded=True,
                ),
            ]
            harness._loaded_tool_names = sorted(
                set(harness._loaded_tool_names)
                | set(external_tools.loaded_tool_names)
                | set(drawio_client.tool_names)
                | {getattr(tool, "name", str(tool)) for tool in build_tools}
            )
            harness._agent = harness._build_root_agent(harness._model)
            return harness
        except Exception:
            await harness.aclose()
            raise

    def _build_root_agent(self, model: str) -> Any:
        return self._backend.build_agent(
            model=model,
            subagents=self._subagents,
            system_prompt=self._system_prompt,
            tools=self._get_guarded_tools(),
            checkpointer=self._runtime.checkpointer,
            state_schema=AgentState,
        )

    def _get_guarded_tools(self) -> list[Any]:
        return [
            self._guard.wrap_tool_with_guard(tool, interrupt_fn=interrupt)
            for tool in self._tools
        ]

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError(
                "Initialize the harness with `await CodingAgentHarness.create(...)`."
            )
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
    def loaded_tool_names(self) -> list[str]:
        return list(self._loaded_tool_names)

    @property
    def mcp_server_status(self) -> dict[str, str]:
        return {**self._external_tools.server_status, **self._additional_mcp_status}

    def create_repair_graph(
        self,
        agent: Any | None = None,
        *,
        max_repair_rounds: int = 3,
        pre_run: Any = None,
    ) -> Any:
        """Create a bounded repair controller around an agent graph."""
        selected_agent = agent if agent is not None else self._agent
        if selected_agent is None:
            raise RuntimeError(
                "Initialize the harness with `await CodingAgentHarness.create(...)`."
            )
        return build_repair_graph(
            selected_agent,
            checkpointer=self._runtime.checkpointer,
            max_repair_rounds=max_repair_rounds,
            guard=self._guard,
            pre_run=pre_run,
            planner=self._planner_agent
            or make_planning_subagent(root_dir=self._root_dir, guard=self._guard),
            root_dir=self._root_dir,
        )

    def build_agent_for_model(self, model: str) -> Any:
        """Build the full planning/design/build agent graph for a model."""
        chosen_model = normalize_model_identifier(model)
        if chosen_model == self._model:
            if self._agent is None:
                raise RuntimeError(
                    "Initialize the harness with `await CodingAgentHarness.create(...)`."
                )
            return self._agent

        return self._build_root_agent(chosen_model)

    def _conversation_history(self, thread_id: str, user_request: str) -> str:
        """Format prior persisted turns, excluding the request currently being run."""
        messages = self._sessions.get_messages(thread_id)
        if (
            messages
            and messages[-1].get("sender") == "user"
            and messages[-1].get("text") == user_request
        ):
            messages = messages[:-1]
        recent_messages = messages[-20:]
        return "\n".join(
            f"{message.get('sender', 'unknown').capitalize()}: {message.get('text', '')}"
            for message in recent_messages
            if message.get("text")
        )

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
        """Return a narrowly matched instruction pattern, ignoring code literals."""
        prose = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
        prose = re.sub(r"`[^`]*`", "", prose)
        for label, pattern in _INJECTION_PATTERNS:
            if pattern.search(prose):
                return label
        return None

    async def run(
        self, user_request: str, thread_id: str, model: str | None = None
    ) -> dict[str, Any]:
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
            if not self._sessions.has_session(thread_id):
                title_preview = user_request.strip().split("\n")[0][:40]
                self._sessions.create_session(
                    title=title_preview or f"Session {thread_id[:8]}",
                    session_id=thread_id,
                )
            self._sessions.update_session(thread_id, status="blocked")
            await self._runtime.persist_session_to_memory(thread_id)
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
            self._sessions.create_session(
                title=title_preview or f"Session {thread_id[:8]}", session_id=thread_id
            )

        messages = self._sessions.get_messages(thread_id)
        if not messages or (
            messages[-1].get("sender") != "user"
            or messages[-1].get("text") != user_request
        ):
            self._sessions.save_message(thread_id, sender="user", text=user_request)
        self._sessions.update_session(thread_id, status="running")
        await self._runtime.persist_session_to_memory(thread_id)

        chosen_model = normalize_model_identifier(model) if model else self._model
        agent_to_invoke = self.build_agent_for_model(chosen_model)

        async def prepare_agent_run(state: dict[str, Any]) -> None:
            await generate_agents_md(
                user_request=state["user_request"],
                root_dir=self._root_dir,
                model=chosen_model,
            )

        execution_graph = self.create_repair_graph(
            agent_to_invoke,
            pre_run=prepare_agent_run,
        )
        self._execution_graphs[thread_id] = execution_graph
        try:
            result = await execution_graph.ainvoke(
                {
                    "user_request": user_request,
                    "conversation_history": self._conversation_history(
                        thread_id, user_request
                    ),
                    "execution_thread_id": f"{thread_id}:execution:{uuid.uuid4()}",
                    "repair_round": 0,
                    "max_repair_rounds": 3,
                },
                config=config,
            )
        except asyncio.CancelledError:
            self._execution_graphs.pop(thread_id, None)
            self._sessions.update_session(thread_id, status="interrupted")
            await self._runtime.persist_session_to_memory(thread_id)
            raise
        except Exception:
            self._sessions.update_session(thread_id, status="failed")
            await self._runtime.persist_session_to_memory(thread_id)
            self._execution_graphs.pop(thread_id, None)
            raise

        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        final_status = result.get("status", "complete")
        self._sessions.update_session(thread_id, status=final_status)
        await self._runtime.persist_session_to_memory(thread_id)
        self._execution_graphs.pop(thread_id, None)
        return {
            "status": final_status,
            "output": result.get("final_output", "Task completed."),
            "result": result,
        }

    async def resume(
        self, thread_id: str, approved: bool | None = None
    ) -> dict[str, Any]:
        """Resume a pending approval or recover a checkpointed run after restart."""
        config = {"configurable": {"thread_id": thread_id}}
        graph = self._execution_graphs.get(thread_id)
        if graph is None:

            async def prepare_recovered_run(state: dict[str, Any]) -> None:
                await generate_agents_md(
                    user_request=state["user_request"],
                    root_dir=self._root_dir,
                    model=self._model,
                )

            graph = self.create_repair_graph(
                agent=self._agent,
                pre_run=prepare_recovered_run,
            )
            self._execution_graphs[thread_id] = graph
        snapshot = await graph.aget_state(config)
        if snapshot is None or not snapshot.values:
            raise ValueError(f"No checkpointed run exists for thread {thread_id}.")
        self._sessions.update_session(thread_id, status="running")
        await self._runtime.persist_session_to_memory(thread_id)
        try:
            if approved is None:
                result = await graph.ainvoke(None, config=config)
            else:
                result = await graph.ainvoke(
                    Command(resume={"approved": approved}),
                    config=config,
                )
        except asyncio.CancelledError:
            self._sessions.update_session(thread_id, status="interrupted")
            await self._runtime.persist_session_to_memory(thread_id)
            raise
        except Exception:
            self._sessions.update_session(thread_id, status="failed")
            await self._runtime.persist_session_to_memory(thread_id)
            self._execution_graphs.pop(thread_id, None)
            raise
        if isinstance(result, dict) and result.get("__interrupt__"):
            self._sessions.update_session(thread_id, status="awaiting_approval")
            await self._runtime.persist_session_to_memory(thread_id)
            return {"status": "awaiting_approval", "interrupt": result["__interrupt__"]}

        final_status = (
            result.get("status", "complete") if isinstance(result, dict) else "complete"
        )
        self._sessions.update_session(thread_id, status=final_status)
        await self._runtime.persist_session_to_memory(thread_id)
        self._execution_graphs.pop(thread_id, None)
        return {
            "status": final_status,
            "output": result.get("final_output", "Task completed.")
            if isinstance(result, dict)
            else "Task completed.",
            "result": result,
        }

    async def get_state(self, thread_id: str) -> dict[str, Any] | None:
        """Retrieves the current state of a thread from checkpointer."""
        config = {"configurable": {"thread_id": thread_id}}
        graph = self._execution_graphs.get(thread_id)
        if graph is None:
            graph = self.create_repair_graph(agent=self._agent)
        snapshot = await graph.aget_state(config)
        return snapshot.values if snapshot else None

    async def stream(self, thread_id: str) -> AsyncIterator[Any]:
        """Streams updates for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self._agent.astream(None, config=config):
            yield chunk

    async def aclose(self) -> None:
        """Close every MCP and database session owned by the harness."""
        await self._external_tools.close_all()
        clients = list(self._mcp_clients)
        if self._drawio_client is not None:
            clients.append(self._drawio_client)
        if clients:
            results = await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Failed to close an MCP session: %s", result)
        await self._runtime.aclose()
