"""
interfaces/api_server.py - FastAPI service providing HTTP & SSE surface over CodingAgentHarness.
"""

import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel

from agent.core import CodingAgentHarness
from tools.github_oauth import vault, build_authorization_url, exchange_code_for_token

app = FastAPI(title="Coding Agent Harness API", version="0.1.0")
harness: CodingAgentHarness | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = None
    session_id: str | None = None


class RunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    approved: bool


@app.on_event("startup")
async def startup_event() -> None:
    global harness
    harness = CodingAgentHarness(root_dir=".", tools=[])


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Multi-Session Management (Codex / Antigravity style)
# ---------------------------------------------------------------------------



@app.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    """List all chat sessions with their metadata, ordered newest first."""
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    from dataclasses import asdict
    return [asdict(s) for s in harness.sessions.list_sessions()]


@app.post("/sessions")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    """Create a new dedicated chat session thread."""
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    from dataclasses import asdict
    session = harness.sessions.create_session(title=req.title, session_id=req.session_id)
    await harness.runtime.persist_session_to_memory(session.id)
    return asdict(session)


@app.get("/sessions/{thread_id}")
async def get_session_details(thread_id: str) -> dict[str, Any]:
    """Retrieve metadata and checkpointed state/messages for a session."""
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    from dataclasses import asdict
    meta = harness.sessions.get_session(thread_id)
    state = await harness.get_state(thread_id)
    return {
        "session": asdict(meta) if meta else None,
        "state": state,
    }


@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str) -> dict[str, Any]:
    """Delete a chat session."""
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    deleted = harness.sessions.delete_session(thread_id)
    return {"thread_id": thread_id, "deleted": deleted}


@app.post("/runs")
async def start_run(req: RunRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())
    result = await harness.run(req.user_request, thread_id=thread_id)
    return {"thread_id": thread_id, **result}


@app.get("/runs/{thread_id}")
async def get_run_status(thread_id: str) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    state = await harness.get_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return state


@app.post("/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    result = await harness.resume(thread_id, approved=req.approved)
    return {"thread_id": thread_id, **result}


@app.get("/runs/{thread_id}/stream")
async def stream_run(thread_id: str) -> StreamingResponse:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    async def event_generator():
        async for chunk in harness.stream(thread_id):
            yield f"data: {str(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/webhooks/github")
async def github_webhook(payload: dict[str, Any]) -> dict[str, str]:
    # Webhook handler for CI status or PR updates
    return {"status": "received"}


# ---------------------------------------------------------------------------
# GitHub OAuth 2.0 — Authorization Code flow
# ---------------------------------------------------------------------------

@app.get("/auth/github")
async def github_connect(
    user_id: str = Query(..., description="Opaque user/session identifier"),
) -> dict[str, str]:
    """Step 1 — Return the GitHub consent-screen URL.

    The frontend redirects the user to the returned ``authorize_url``.
    GitHub then redirects back to /auth/github/callback with a one-time code.
    """
    try:
        url = build_authorization_url(user_id=user_id)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"authorize_url": url, "user_id": user_id}


@app.get("/auth/github/callback")
async def github_oauth_callback(
    code: str = Query(..., description="One-time code from GitHub"),
    state: str = Query(..., description="user_id echoed back as OAuth state"),
) -> dict[str, str]:
    """Step 2 — Exchange the temporary code for an access token.

    GitHub calls this endpoint automatically after the user approves the
    consent screen. The token is stored in the in-memory vault; the user_id
    is recovered from the ``state`` query parameter.
    """
    user_id = state
    try:
        token = await exchange_code_for_token(code=code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {exc}")

    vault.set(user_id, "github", token)
    return {"status": "connected", "user_id": user_id}


@app.get("/auth/github/status/{user_id}")
async def github_auth_status(user_id: str) -> dict[str, Any]:
    """Return whether a GitHub token is stored for this user."""
    return {"user_id": user_id, "connected": vault.has(user_id, "github")}


@app.delete("/auth/github/revoke/{user_id}")
async def github_revoke(user_id: str) -> dict[str, str]:
    """Remove the stored GitHub token for this user (logout / re-auth)."""
    vault.revoke(user_id, "github")
    return {"status": "revoked", "user_id": user_id}
