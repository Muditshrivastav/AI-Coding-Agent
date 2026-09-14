"""
agent/session_manager.py - Multi-session manager for chat and task threads.

Provides Codex / Antigravity-style session lifecycle management:
- Create new named or auto-titled chat sessions
- List all active and archived sessions with summary metadata
- Track message count, active stage, and timestamp per session
- Delete or archive existing sessions
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMetadata:
    id: str
    title: str
    created_at: str
    updated_at: str
    stage: str = "planning"
    message_count: int = 0
    status: str = "active"  # "active" | "awaiting_approval" | "completed" | "archived"


class SessionManager:
    """Manages chat session lifecycle and metadata registry."""

    def __init__(self, root_dir: str = ".") -> None:
        self._root_dir = Path(root_dir)
        self._storage_dir = self._root_dir / "harness" / "sessions"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._storage_dir / "sessions.json"
        self._sessions: dict[str, SessionMetadata] = {}
        self._load()

    def _load(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    meta = SessionMetadata(**item)
                    self._sessions[meta.id] = meta
        except Exception:
            self._sessions = {}

    def _save(self) -> None:
        try:
            items = [asdict(meta) for meta in self._sessions.values()]
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
        except Exception:
            pass

    def create_session(self, title: str | None = None, session_id: str | None = None) -> SessionMetadata:
        """Create a new chat session with a unique ID."""
        sid = session_id or str(uuid.uuid4())
        now = _utc_now_iso()
        session_title = title.strip() if title else f"Session {sid[:8]}"
        meta = SessionMetadata(
            id=sid,
            title=session_title,
            created_at=now,
            updated_at=now,
        )
        self._sessions[sid] = meta
        self._save()
        return meta

    def get_session(self, session_id: str) -> SessionMetadata | None:
        """Retrieve session metadata by ID."""
        return self._sessions.get(session_id)

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def list_sessions(self) -> list[SessionMetadata]:
        """List all sessions ordered by updated_at descending."""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        stage: str | None = None,
        status: str | None = None,
        increment_messages: bool = False,
    ) -> SessionMetadata | None:
        """Update session activity and status."""
        meta = self._sessions.get(session_id)
        if not meta:
            return None

        meta.updated_at = _utc_now_iso()
        if title:
            meta.title = title
        if stage:
            meta.stage = stage
        if status:
            meta.status = status
        if increment_messages:
            meta.message_count += 1

        self._save()
        return meta

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from the registry."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            return True
        return False
