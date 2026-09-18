"""
frameworks/supabase_memory.py - Persistent cross-session memory via asyncpg / PostgreSQL.
"""

import json
from typing import Any
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()


class SupabaseMemory:
    """Manages cross-session key-value memory persisted in Supabase / Postgres via asyncpg."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = (
            dsn
            or os.getenv("supabase_uri")
            or os.getenv("supabase_url")
        )
        if not self._dsn:
            raise ValueError(
                "No PostgreSQL connection URI provided. Please set 'supabase_uri' or 'DATABASE_URL' in .env or pass dsn."
            )
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # 1. Create table with expanded schema if not exists
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    project_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value JSONB NOT NULL,
                    session_id TEXT,
                    memory_type TEXT DEFAULT 'general',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (project_id, key)
                );
                """
            )
            # 2. Add columns if table already existed with older 3-field schema
            await conn.execute(
                """
                ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS session_id TEXT;
                ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS memory_type TEXT DEFAULT 'general';
                ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
                ALTER TABLE agent_memory ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();
                CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory (session_id);
                CREATE INDEX IF NOT EXISTS idx_agent_memory_type ON agent_memory (project_id, memory_type);
                """
            )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def save(
        self,
        project_id: str,
        key: str,
        value: dict[str, Any],
        session_id: str | None = None,
        memory_type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Saves or updates a memory record with optional session_id, type, and metadata."""
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memory (project_id, key, value, session_id, memory_type, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, now())
                ON CONFLICT (project_id, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    session_id = COALESCE(EXCLUDED.session_id, agent_memory.session_id),
                    memory_type = EXCLUDED.memory_type,
                    metadata = EXCLUDED.metadata,
                    updated_at = now();
                """,
                project_id,
                key,
                json.dumps(value),
                session_id,
                memory_type,
                json.dumps(metadata or {}),
            )

    async def load(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Loads value from agent_memory by project_id and key."""
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM agent_memory WHERE project_id = $1 AND key = $2",
                project_id,
                key,
            )
            return json.loads(row["value"]) if row else None

    async def load_record(self, project_id: str, key: str) -> dict[str, Any] | None:
        """Loads the full memory record including session_id, metadata, and timestamps."""
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT project_id, key, value, session_id, memory_type, metadata, created_at, updated_at
                FROM agent_memory
                WHERE project_id = $1 AND key = $2
                """,
                project_id,
                key,
            )
            if not row:
                return None
            return {
                "project_id": row["project_id"],
                "key": row["key"],
                "value": json.loads(row["value"]) if isinstance(row["value"], str) else row["value"],
                "session_id": row["session_id"],
                "memory_type": row["memory_type"],
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }

    # ----------------------------------------------------------------------
    # Session-specific Memory Persistence
    # ----------------------------------------------------------------------

    async def save_session(
        self,
        project_id: str,
        session_id: str,
        session_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stores a full session record into agent_memory under memory_type='session'."""
        key = f"session:{session_id}"
        await self.save(
            project_id=project_id,
            key=key,
            value=session_data,
            session_id=session_id,
            memory_type="session",
            metadata=metadata,
        )

    async def get_session(self, project_id: str, session_id: str) -> dict[str, Any] | None:
        """Retrieves session data stored under memory_type='session'."""
        key = f"session:{session_id}"
        return await self.load(project_id=project_id, key=key)

    async def list_sessions(self, project_id: str) -> list[dict[str, Any]]:
        """Lists all stored sessions for a project ordered newest first."""
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT project_id, session_id, key, value, metadata, created_at, updated_at
                FROM agent_memory
                WHERE project_id = $1 AND memory_type = 'session'
                ORDER BY updated_at DESC
                """,
                project_id,
            )
            result = []
            for row in rows:
                val = json.loads(row["value"]) if isinstance(row["value"], str) else row["value"]
                meta = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
                result.append({
                    "project_id": row["project_id"],
                    "session_id": row["session_id"],
                    "key": row["key"],
                    "session_data": val,
                    "metadata": meta,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                })
            return result

    async def delete_session(self, project_id: str, session_id: str) -> bool:
        """Deletes all session and session-related memories for a session_id."""
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM agent_memory
                WHERE project_id = $1 AND (session_id = $2 OR key = $3)
                """,
                project_id,
                session_id,
                f"session:{session_id}",
            )
            return "DELETE 0" not in result
