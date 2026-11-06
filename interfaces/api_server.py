"""
interfaces/api_server.py - FastAPI service providing HTTP & SSE surface over CodingAgentHarness.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel

# Ensure repository root is on sys.path even when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force the process to work within the project root to avoid CWD issues
os.chdir(PROJECT_ROOT)

from fastapi.middleware.cors import CORSMiddleware
from frameworks.vscode_workspace import resolve_workspace_root
from tools.github_oauth import vault, build_authorization_url, exchange_code_for_token

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO)

harness = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global harness
    logger.info("AGENT_ROOT_DIR = %s", os.environ.get("AGENT_ROOT_DIR", "<not set>"))
    logger.info("VSCODE_CWD = %s", os.environ.get("VSCODE_CWD", "<not set>"))
    try:
        from agent.core import CodingAgentHarness

        root_dir = resolve_workspace_root()
        logger.info("Agent workspace root resolved to: %s", root_dir)
        harness = await CodingAgentHarness.create(root_dir=root_dir, tools=None)
        logger.info("CodingAgentHarness initialized successfully.")
    except Exception:
        logger.exception("Harness startup failed.")

    try:
        yield
    finally:
        if harness is not None:
            await harness.aclose()
            harness = None


app = FastAPI(
    title="Coding Agent Harness API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreateSessionRequest(BaseModel):
    title: str | None = None
    session_id: str | None = None


class RunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None
    model: str | None = None


class ResumeRequest(BaseModel):
    approved: bool | None = None


@app.get("/health")
async def health_check() -> dict[str, Any]:
    global harness
    root = harness.root_dir if harness else resolve_workspace_root()
    return {
        "status": "ok" if harness else "degraded",
        "workspace_root": root,
        "workspace_name": os.path.basename(root) or root,
        "agent_root_env": os.environ.get("AGENT_ROOT_DIR", ""),
        "loaded_tools": harness.loaded_tool_names if harness else [],
        "mcp_servers": harness.mcp_server_status if harness else {},
        "sandbox_mode": harness._backend.sandbox_mode if harness else None,
    }


@app.get("/tools")
async def list_loaded_tools() -> dict[str, Any]:
    """Expose the tools and MCP servers actually available to the running harness."""
    if harness is None:
        raise HTTPException(status_code=503, detail="Harness not initialized.")
    return {
        "tools": harness.loaded_tool_names,
        "mcp_servers": harness.mcp_server_status,
    }


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """Return all supported LLM models and default active model."""
    from agent.core import DEFAULT_MODEL, SUPPORTED_MODELS
    global harness
    current_model = harness.model if harness else DEFAULT_MODEL
    return {
        "current_model": current_model,
        "models": [
            {
                "id": m["id"],
                "name": m["name"],
                "label": m["label"],
                "provider": m["provider"],
            }
            for m in SUPPORTED_MODELS.values()
        ],
    }


class SetWorkspaceRequest(BaseModel):
    root_dir: str


@app.get("/workspace")
async def get_workspace() -> dict[str, Any]:
    """Return the currently active workspace root and env diagnostics."""
    global harness
    root = harness.root_dir if harness else resolve_workspace_root()
    return {
        "workspace_root": root,
        "workspace_name": os.path.basename(root) or root,
        "agent_root_env": os.environ.get("AGENT_ROOT_DIR", ""),
        "vscode_cwd_env": os.environ.get("VSCODE_CWD", ""),
        "process_cwd": os.getcwd(),
    }


@app.post("/workspace/set")
async def set_workspace(req: SetWorkspaceRequest) -> dict[str, Any]:
    """Hot-swap the agent workspace root without restarting uvicorn.

    Sets AGENT_ROOT_DIR in the process environment so resolve_workspace_root()
    picks it up, then reinitialises CodingAgentHarness with the new path.
    Existing sessions remain in the session store but new runs will use
    the new root_dir.
    """
    global harness
    import os as _os
    new_root = _os.path.abspath(req.root_dir)
    if not _os.path.isdir(new_root):
        raise HTTPException(status_code=400, detail=f"Directory not found: {new_root}")

    if any(not task.done() for task in active_run_tasks.values()):
        raise HTTPException(status_code=409, detail="Stop active runs before changing workspaces.")

    old_harness = harness
    old_root = _os.environ.get("AGENT_ROOT_DIR")
    old_vscode_cwd = _os.environ.get("VSCODE_CWD")
    # Propagate into process env so reload-triggered workers also see it
    _os.environ["AGENT_ROOT_DIR"] = new_root
    _os.environ["VSCODE_CWD"] = ""  # clear interfering var

    try:
        from agent.core import CodingAgentHarness
        new_harness = await CodingAgentHarness.create(root_dir=new_root, tools=None)
    except Exception as exc:
        if old_root is None:
            _os.environ.pop("AGENT_ROOT_DIR", None)
        else:
            _os.environ["AGENT_ROOT_DIR"] = old_root
        if old_vscode_cwd is None:
            _os.environ.pop("VSCODE_CWD", None)
        else:
            _os.environ["VSCODE_CWD"] = old_vscode_cwd
        import traceback
        logger.error(f"❌ Workspace set failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))

    harness = new_harness
    if old_harness is not None:
        try:
            await old_harness.aclose()
        except Exception:
            logger.exception("Failed to close the previous workspace harness.")
    logger.info(f"🔄 Workspace hot-swapped → {new_root}")
    return {
        "status": "ok",
        "workspace_root": new_root,
        "workspace_name": _os.path.basename(new_root) or new_root,
    }



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
    messages = harness.sessions.get_messages(thread_id)
    return {
        "session": asdict(meta) if meta else None,
        "state": state,
        "messages": messages,
    }


@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str) -> dict[str, Any]:
    """Delete a chat session."""
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    deleted = harness.sessions.delete_session(thread_id)
    return {"thread_id": thread_id, "deleted": deleted}


active_run_tasks: dict[str, asyncio.Task] = {}


async def _run_in_background(
    thread_id: str,
    user_request: str,
    model: str | None,
) -> None:
    """Run one agent task without holding the initiating HTTP request open."""
    active_harness = harness
    if active_harness is None:
        return
    try:
        result = await active_harness.run(
            user_request,
            thread_id=thread_id,
            model=model,
        )
        if result.get("status") != "awaiting_approval" and result.get("output"):
            active_harness.sessions.save_message(
                thread_id,
                sender="agent",
                text=str(result["output"]),
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Background run failed for thread %s.", thread_id)
    finally:
        if active_run_tasks.get(thread_id) is asyncio.current_task():
            active_run_tasks.pop(thread_id, None)


@app.post("/runs")
async def start_run(req: RunRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())
    if thread_id in active_run_tasks and not active_run_tasks[thread_id].done():
        raise HTTPException(status_code=409, detail="A run is already active for this thread.")
    if not harness.sessions.has_session(thread_id):
        title_preview = req.user_request.strip().splitlines()
        harness.sessions.create_session(
            title=title_preview[0][:40] if title_preview else None,
            session_id=thread_id,
        )
    existing_messages = harness.sessions.get_messages(thread_id)
    if not existing_messages or (
        existing_messages[-1].get("sender") != "user"
        or existing_messages[-1].get("text") != req.user_request
    ):
        harness.sessions.save_message(thread_id, sender="user", text=req.user_request)
    harness.sessions.update_session(thread_id, status="running")
    task = asyncio.create_task(
        _run_in_background(thread_id, req.user_request, req.model),
        name=f"agent-run-{thread_id}",
    )
    active_run_tasks[thread_id] = task
    return {"thread_id": thread_id, "status": "running"}

import json as _json


class StreamRunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None
    model: str | None = None


def _interrupt_details(interrupts: Any) -> dict[str, Any]:
    if not interrupts:
        return {}
    first = interrupts[0]
    raw = getattr(first, "value", first) if not isinstance(first, dict) else first
    return raw if isinstance(raw, dict) else {}


@app.post("/runs/stream")
async def start_run_stream(req: StreamRunRequest) -> StreamingResponse:
    """Run the agent and stream SSE events for tool calls + final response.

    Event types emitted:
      data: {"type": "tool_call",   "tool": "<name>", "input": "<summary>", "id": "<id>"}
      data: {"type": "tool_result", "tool": "<name>", "output": "<summary>", "id": "<id>", "lines": N}
      data: {"type": "text_delta",  "text": "<chunk>"}
      data: {"type": "done",        "output": "<full text>", "status": "complete|needs_attention|stopped|error"}
    """
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())
    harness.sessions.save_message(thread_id, sender="user", text=req.user_request)

    async def event_generator():
        # Local capture to satisfy type checker across closure boundary
        active_harness = harness
        if active_harness is None:
            return
        current_task = asyncio.current_task()
        if current_task is not None:
            active_run_tasks[thread_id] = current_task

        try:
            config = {"configurable": {"thread_id": thread_id}}
            if not active_harness.sessions.has_session(thread_id):
                title_preview = req.user_request.strip().split("\n")[0][:40]
                active_harness.sessions.create_session(title=title_preview or f"Session {thread_id[:8]}", session_id=thread_id)
            conversation_history = active_harness._conversation_history(thread_id, req.user_request)

            from agent.core import normalize_model_identifier
            from frameworks.agents_md_writer import generate_agents_md
            chosen_model = normalize_model_identifier(req.model) if req.model else active_harness._model
            logger.info("Starting streamed run with model %s.", chosen_model)
            agent_to_invoke = await active_harness.get_agent_for_model(chosen_model)

            async def prepare_agent_run(state: dict[str, Any]) -> None:
                await generate_agents_md(
                    user_request=state["user_request"],
                    root_dir=active_harness.root_dir,
                    model=chosen_model,
                )

            execution_graph = active_harness.create_repair_graph(
                agent_to_invoke,
                pre_run=prepare_agent_run,
                planner=active_harness._planners_by_model[chosen_model],
            )
            active_harness._execution_graphs[thread_id] = execution_graph
            result_state: dict[str, Any] = {}
            try:
                async for mode, chunk in execution_graph.astream(
                    {
                        "user_request": req.user_request,
                        "conversation_history": conversation_history,
                        "execution_thread_id": f"{thread_id}:execution:{uuid.uuid4()}",
                        "model_id": chosen_model,
                        "repair_round": 0,
                        "max_repair_rounds": 3,
                    },
                    config=config,
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        if isinstance(chunk, dict):
                            event = _json.dumps(chunk)
                            yield f"data: {event}\n\n"
                    elif mode == "values" and isinstance(chunk, dict):
                        result_state = chunk
            except Exception as stream_error:
                if type(stream_error).__name__ in ("GraphInterrupt", "NodeInterrupt"):
                    payload = _interrupt_details(getattr(stream_error, "interrupts", None))
                    active_harness.sessions.update_session(thread_id, status="awaiting_approval")
                    await active_harness._runtime.persist_session_to_memory(thread_id)
                    interrupt_event = _json.dumps({
                        "type": "interrupt",
                        "description": payload.get("description") or payload.get("signature", "Action requires approval"),
                        "action": payload.get("signature", "Shell / File modification"),
                        "payload": payload,
                    })
                    yield f"data: {interrupt_event}\n\n"
                    logger.info(f"HITL interrupt raised for thread {thread_id}: {payload}")
                    return
                raise

            if result_state.get("__interrupt__"):
                payload = _interrupt_details(result_state["__interrupt__"])
                active_harness.sessions.update_session(thread_id, status="awaiting_approval")
                await active_harness._runtime.persist_session_to_memory(thread_id)
                interrupt_event = _json.dumps({
                    "type": "interrupt",
                    "description": payload.get("description") or payload.get("signature", "Action requires approval"),
                    "action": payload.get("signature", "Shell / File modification"),
                    "payload": payload,
                })
                yield f"data: {interrupt_event}\n\n"
                logger.info(f"HITL interrupt raised for thread {thread_id}: {payload}")
                return

            output_text = result_state.get("final_output", "Task completed.")
            if not isinstance(output_text, str) or not output_text.strip():
                output_text = "Task completed."

            final_status = result_state.get("status", "complete")
            active_harness.sessions.update_session(thread_id, status=final_status)
            await active_harness._runtime.persist_session_to_memory(thread_id)
            active_harness.sessions.save_message(thread_id, sender="agent", text=output_text)
            active_harness._execution_graphs.pop(thread_id, None)
            done_ev = _json.dumps({
                "type": "done",
                "output": output_text,
                "status": final_status,
            })
            yield f"data: {done_ev}\n\n"

        except asyncio.CancelledError:
            active_harness.sessions.update_session(thread_id, status="interrupted")
            await active_harness._runtime.persist_session_to_memory(thread_id)
            raise
        except Exception as exc:
            import traceback
            logger.error(f"Stream run error: {exc}\n{traceback.format_exc()}")
            active_harness._execution_graphs.pop(thread_id, None)
            done_ev = _json.dumps({"type": "done", "output": f"⚠️ Agent error: {exc}", "status": "error"})
            yield f"data: {done_ev}\n\n"
        finally:
            active_run_tasks.pop(thread_id, None)
            session = active_harness.sessions.get_session(thread_id)
            if not session or session.status != "awaiting_approval":
                active_harness._execution_graphs.pop(thread_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/runs/{thread_id}/stop")
async def stop_run(thread_id: str) -> dict[str, Any]:
    """Stops/cancels an in-flight agent model execution."""
    task = active_run_tasks.get(thread_id)
    if task and not task.done():
        task.cancel()
        return {"thread_id": thread_id, "status": "stopping", "message": "Execution cancellation requested."}
    return {"thread_id": thread_id, "status": "not_running", "message": "No active task found for thread."}


@app.get("/runs/{thread_id}")
async def get_run_status(thread_id: str) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    active_harness = harness
    state = await active_harness.get_state(thread_id) or {}
    messages = active_harness.sessions.get_messages(thread_id)
    session = active_harness.sessions.get_session(thread_id)
    if "messages" not in state or not state["messages"]:
        state["messages"] = messages
    return {
        **state,
        "thread_id": thread_id,
        "status": session.status if session else state.get("status", "unknown"),
        "current_task": state.get("current_task"),
        "progress": state.get("tasks", []),
        "pending_approval": bool(session and session.status == "awaiting_approval"),
    }


@app.post("/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    active_harness = harness
    if thread_id in active_run_tasks and not active_run_tasks[thread_id].done():
        raise HTTPException(status_code=409, detail="A run is already active for this thread.")
    session = active_harness.sessions.get_session(thread_id)
    if session and session.status == "failed":
        raise HTTPException(
            status_code=409,
            detail=(
                "This run has failed and was stopped. It cannot be resumed; "
                "start a new run to try again."
            ),
        )
    if session and session.status == "awaiting_approval" and req.approved is None:
        raise HTTPException(
            status_code=400,
            detail="An approval decision is required to resume this run.",
        )
    try:
        result = await active_harness.resume(thread_id, approved=req.approved)
    except ValueError as exc:
        active_harness._guard.log_failure(
            stage="resume",
            category="unavailable-checkpoint",
            description=f"Thread {thread_id}: {exc}",
            resolution="Resume stopped; start a new run instead of retrying this checkpoint.",
        )
        logger.warning("Run %s cannot be resumed: %s", thread_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        active_harness._guard.log_failure(
            stage="resume",
            category=type(exc).__name__,
            description=f"Thread {thread_id}: {exc}",
            resolution="Resume stopped and the session was marked failed; start a new run.",
        )
        logger.exception("Failed to resume run %s.", thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Resume failed ({type(exc).__name__}): {exc}",
        ) from exc
    if result.get("status") != "awaiting_approval" and result.get("output"):
        active_harness.sessions.save_message(thread_id, sender="agent", text=result["output"])
    return {"thread_id": thread_id, **result}


@app.get("/runs/{thread_id}/stream")
async def stream_run(thread_id: str) -> StreamingResponse:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    active_harness = harness
    async def event_generator():
        async for chunk in active_harness.stream(thread_id):
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


@app.get("/workspace/files")
async def get_workspace_files(path: str = "") -> dict[str, Any]:
    """Return directory tree listing for workspace viewer."""
    import os
    global harness
    base_dir = os.path.abspath(harness.root_dir if harness else ".")
    target_dir = os.path.abspath(os.path.join(base_dir, path))
    if not target_dir.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Forbidden")

    entries = []
    try:
        for entry in os.scandir(target_dir):
            if entry.name.startswith((".", "__pycache__", "venv", "node_modules")):
                continue
            entries.append({
                "name": entry.name,
                "path": os.path.relpath(entry.path, base_dir).replace("\\", "/"),
                "is_dir": entry.is_dir(),
                "size": entry.stat().st_size if not entry.is_dir() else None,
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return {"files": entries, "current_path": path}


@app.get("/workspace/file")
async def get_file_content(path: str = Query(..., description="Relative file path")) -> dict[str, Any]:
    """Read file content for IDE code viewer."""
    import os
    global harness
    base_dir = os.path.abspath(harness.root_dir if harness else ".")
    full_path = os.path.abspath(os.path.join(base_dir, path))
    if not full_path.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not os.path.exists(full_path) or os.path.isdir(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(100_000)
        return {"path": path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
