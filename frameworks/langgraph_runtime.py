"""
frameworks/langgraph_runtime.py - LangGraph runtime and checkpointer management.

Main Purpose:
1. Manages LangGraph execution state and checkpointers (e.g. MemorySaver).
2. Integrates with SessionManager and SupabaseMemory:
   - Tracks multi-session metadata and thread configs via SessionManager.
   - Bridges cross-session persistent state and key-value memory via SupabaseMemory (PostgreSQL / asyncpg),
     keeping LangGraph checkpointers decoupled from external asyncpg connections.
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver

from agent.session_manager import SessionManager, SessionMetadata
from frameworks.supabase_memory import SupabaseMemory

load_dotenv()


class LangGraphRuntime:
    """Manages LangGraph checkpoints, session thread mapping, and memory integration.

    Integrates SessionManager and SupabaseMemory so that each thread ID possesses
    both active graph state in checkpointer and durable cross-session persistent storage in Supabase.
    """

    def __init__(
        self,
        checkpointer: Any = None,
        session_manager: SessionManager | None = None,
        supabase_memory: SupabaseMemory | None = None,
        root_dir: str = ".",
    ) -> None:
        self._checkpointer = checkpointer or MemorySaver()
        self._session_manager = session_manager or SessionManager(root_dir=root_dir)

        # Connect cross-session storage via SupabaseMemory (asyncpg)
        if supabase_memory is not None:
            self._supabase_memory: SupabaseMemory | None = supabase_memory
        else:
            db_dsn = (
                os.getenv("supabase_uri")
                or os.getenv("SUPABASE_URI")
                or os.getenv("DATABASE_URL")
                or os.getenv("SUPABASE_DB_URL")
            )
            self._supabase_memory = SupabaseMemory(db_dsn) if db_dsn else None

    @property
    def checkpointer(self) -> Any:
        """The active LangGraph BaseCheckpointSaver instance."""
        return self._checkpointer

    @property
    def session_manager(self) -> SessionManager:
        """The underlying SessionManager instance."""
        return self._session_manager

    @property
    def supabase_memory(self) -> SupabaseMemory | None:
        """The underlying SupabaseMemory instance (if configured with a valid DSN)."""
        return self._supabase_memory

    @classmethod
    def create_memory_saver(cls) -> MemorySaver:
        """Creates an in-memory checkpointer."""
        return MemorySaver()

    # ----------------------------------------------------------------------
    # Multi-session Thread & Config Management
    # ----------------------------------------------------------------------

    def get_thread_config(self, thread_id: str) -> dict[str, Any]:
        """Returns the LangGraph RunnableConfig dictionary configured for this thread_id."""
        return {"configurable": {"thread_id": thread_id}}

    def ensure_session(self, thread_id: str, title: str | None = None) -> SessionMetadata:
        """Ensures that a session entry exists in SessionManager for the given thread_id."""
        session = self._session_manager.get_session(thread_id)
        if not session:
            session = self._session_manager.create_session(title=title, session_id=thread_id)
        return session

    def list_sessions(self) -> list[SessionMetadata]:
        """Lists all registered sessions with metadata (message count, status, stage)."""
        return self._session_manager.list_sessions()

    def update_session_state(
        self,
        thread_id: str,
        *,
        stage: str | None = None,
        status: str | None = None,
        increment_messages: bool = False,
    ) -> SessionMetadata | None:
        """Synchronizes session metadata in SessionManager with LangGraph node execution."""
        return self._session_manager.update_session(
            session_id=thread_id,
            stage=stage,
            status=status,
            increment_messages=increment_messages,
        )

    # ----------------------------------------------------------------------
    # Supabase Memory Thread/Project Persistence Integration
    # ----------------------------------------------------------------------

    async def save_session_memory(
        self,
        session_id: str,
        key: str,
        value: dict[str, Any],
        memory_type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persists session memory to SupabaseMemory with session tracking."""
        if self._supabase_memory:
            await self._supabase_memory.save(
                project_id=session_id,
                key=key,
                value=value,
                session_id=session_id,
                memory_type=memory_type,
                metadata=metadata,
            )

    async def load_session_memory(self, session_id: str, key: str) -> dict[str, Any] | None:
        """Loads session memory from SupabaseMemory."""
        if self._supabase_memory:
            return await self._supabase_memory.load(project_id=session_id, key=key)
        return None

    async def persist_session_to_memory(self, session_id: str) -> None:
        """Persists the SessionMetadata directly to agent_memory as a session record."""
        if not self._supabase_memory:
            return
        meta = self._session_manager.get_session(session_id)
        if meta:
            from dataclasses import asdict
            await self._supabase_memory.save_session(
                project_id=session_id,
                session_id=session_id,
                session_data=asdict(meta),
                metadata={"title": meta.title, "stage": meta.stage, "status": meta.status},
            )
