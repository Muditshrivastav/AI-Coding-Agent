"""
frameworks/supabase_memory.py - Persistent cross-session memory via asyncpg / PostgreSQL.
"""

import json
from typing import Any
import asyncpg


class SupabaseMemory:
    """Manages cross-session key-value memory persisted in Supabase / Postgres via asyncpg."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    project_id TEXT,
                    key TEXT,
                    value JSONB,
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (project_id, key)
                );
                """
            )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def save(self, project_id: str, key: str, value: dict[str, Any]) -> None:
        if self._pool is None:
            await self.init()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memory (project_id, key, value)
                VALUES ($1, $2, $3)
                ON CONFLICT (project_id, key)
                DO UPDATE SET value = $3, updated_at = now();
                """,
                project_id,
                key,
                json.dumps(value),
            )

    async def load(self, project_id: str, key: str) -> dict[str, Any] | None:
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
