"""
interfaces/api_server.py - FastAPI service providing HTTP & SSE surface over CodingAgentHarness.
"""

import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel

import logging
import os
from pathlib import Path
import sys

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

app = FastAPI(title="Coding Agent Harness API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

harness = None


class CreateSessionRequest(BaseModel):
    title: str | None = None
    session_id: str | None = None


class RunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None
    model: str | None = None


class ResumeRequest(BaseModel):
    approved: bool


@app.on_event("startup")
async def startup_event() -> None:
    global harness
    # Log the env var explicitly so it's visible in the uvicorn terminal
    agent_root_env = os.environ.get("AGENT_ROOT_DIR", "<not set>")
    vscode_cwd_env = os.environ.get("VSCODE_CWD", "<not set>")
    logger.info(f"🔑 AGENT_ROOT_DIR = {agent_root_env}")
    logger.info(f"🔑 VSCODE_CWD     = {vscode_cwd_env}")
    try:
        from agent.core import CodingAgentHarness
        root_dir = resolve_workspace_root()
        logger.info(f"🗂️  Agent workspace root resolved to: {root_dir}")
        harness = CodingAgentHarness(root_dir=root_dir, tools=None, sandbox_mode="local")
        logger.info("✅ CodingAgentHarness initialized successfully (sandbox_mode=local).")
    except Exception as exc:
        import traceback
        logger.error(f"❌ Harness startup failed: {exc}")
        logger.error(traceback.format_exc())
        # Leave harness=None so endpoints return 503 instead of crashing


@app.get("/health")
async def health_check() -> dict[str, Any]:
    global harness
    root = harness.root_dir if harness else resolve_workspace_root()
    return {
        "status": "ok" if harness else "degraded",
        "workspace_root": root,
        "workspace_name": os.path.basename(root) or root,
        "agent_root_env": os.environ.get("AGENT_ROOT_DIR", ""),
    }


@app.get("/models")
async def get_models() -> dict[str, Any]:
    """Return all supported LLM models and default active model."""
    from agent.core import SUPPORTED_MODELS
    global harness
    current_model = harness.model if harness else "groq:qwen/qwen3.8-27b"
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

    # Propagate into process env so reload-triggered workers also see it
    _os.environ["AGENT_ROOT_DIR"] = new_root
    _os.environ["VSCODE_CWD"] = ""  # clear interfering var

    try:
        from agent.core import CodingAgentHarness
        harness = CodingAgentHarness(root_dir=new_root, tools=None, sandbox_mode="local")
        logger.info(f"🔄 Workspace hot-swapped → {new_root}")
        return {
            "status": "ok",
            "workspace_root": new_root,
            "workspace_name": _os.path.basename(new_root) or new_root,
        }
    except Exception as exc:
        import traceback
        logger.error(f"❌ Workspace set failed: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))



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


import asyncio

active_run_tasks: dict[str, asyncio.Task] = {}


@app.post("/runs")
async def start_run(req: RunRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    thread_id = req.thread_id or str(uuid.uuid4())

    # Persist the user message immediately into session storage
    harness.sessions.save_message(thread_id, sender="user", text=req.user_request)

    # Run harness.run in an asyncio Task so it can be cancelled on request
    task = asyncio.create_task(
        harness.run(req.user_request, thread_id=thread_id, model=req.model)
    )
    active_run_tasks[thread_id] = task

    try:
        result = await task

        # Extract agent response text from LangGraph AgentState:
        # The result is typically {"status": "complete", "result": AgentState}
        # AgentState has a "messages" list of LangChain message objects.
        output_text = None
        inner = result.get("result") or result  # unwrap if wrapped in {status, result}
        if isinstance(inner, dict):
            msgs = inner.get("messages") or []
            # Walk messages from the end and find the last AI/assistant message
            for m in reversed(msgs):
                role = getattr(m, "type", None) or getattr(m, "role", None) or ""
                if role in ("ai", "assistant"):
                    content = getattr(m, "content", None)
                    if isinstance(content, str) and content.strip():
                        output_text = content.strip()
                        break
                    elif isinstance(content, list):
                        # Content can be a list of blocks [{type: text, text: ...}]
                        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                        joined = " ".join(p for p in parts if p).strip()
                        if joined:
                            output_text = joined
                            break

        if output_text is None:
            # Fallback: legacy output/response keys
            output_text = (
                result.get("output")
                or result.get("response")
                or (str(result.get("result")) if result.get("result") else None)
                or "Task completed."
            )

        if result.get("status") != "awaiting_approval":
            harness.sessions.save_message(thread_id, sender="agent", text=output_text)
        return {"thread_id": thread_id, "output": output_text, **{k: v for k, v in result.items() if k != "result"}}
    except asyncio.CancelledError:
        logger.info(f"🛑 Run on thread {thread_id} was cancelled by user.")
        harness.sessions.save_message(thread_id, sender="agent", text="🛑 Execution stopped by user.")
        return {
            "thread_id": thread_id,
            "status": "stopped",
            "output": "🛑 Execution stopped by user.",
        }
    finally:
        active_run_tasks.pop(thread_id, None)
        if harness is not None:
            session = harness.sessions.get_session(thread_id)
            if not session or session.status != "awaiting_approval":
                harness._execution_graphs.pop(thread_id, None)

import json as _json


class StreamRunRequest(BaseModel):
    user_request: str
    thread_id: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Groq rate-limit helpers
# ---------------------------------------------------------------------------

def _is_groq_rate_limit(exc: Exception) -> bool:
    """Return True when *exc* represents a Groq HTTP 429 rate-limit response.

    Groq's Python SDK wraps the HTTP response as ``groq.RateLimitError`` or as
    a generic ``httpx.HTTPStatusError`` / ``requests.HTTPError`` with status 429.
    We also do a fallback string-match so that any future SDK change is still caught.
    """
    # 1. Exact status_code attribute (groq-sdk, httpx, requests)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True

    # 2. Nested response object (httpx / requests style)
    resp = getattr(exc, "response", None)
    if resp is not None:
        if getattr(resp, "status_code", None) == 429:
            return True

    # 3. Class name contains "RateLimit"
    if "ratelimit" in type(exc).__name__.lower():
        return True

    # 4. Fallback: scan the exception message for "429" or "rate limit"
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True

    return False


def _interrupt_details(interrupts: Any) -> dict[str, Any]:
    if not interrupts:
        return {}
    first = interrupts[0]
    raw = getattr(first, "value", first) if not isinstance(first, dict) else first
    return raw if isinstance(raw, dict) else {}


_OLLAMA_FALLBACKS = [
    {"id": "ollama:gpt-oss:120b-cloud",            "label": "GPT-OSS 120B (Cloud)"},
    {"id": "ollama:gemma4:cloud",                  "label": "Gemma 4 (Cloud)"},
    {"id": "ollama:nvidia-nemotron-super:cloud",   "label": "NVIDIA Nemotron Super (Cloud)"},
]


def _make_rate_limit_event(current_model: str) -> dict:
    """Build the SSE payload emitted when Groq returns a 429."""
    return {
        "type": "rate_limit",
        "provider": "groq",
        "current_model": current_model,
        "message": (
            "⚡ Groq rate limit hit (HTTP 429). You have exhausted your Groq token quota. "
            "Switch to an Ollama model to continue without interruption."
        ),
        "fallbacks": _OLLAMA_FALLBACKS,
    }


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
            await generate_agents_md(
                user_request=req.user_request,
                root_dir=active_harness.root_dir,
                model=chosen_model,
            )

            agent_to_invoke = active_harness.build_agent_for_model(chosen_model)

            execution_graph = active_harness.create_repair_graph(agent_to_invoke)
            active_harness._execution_graphs[thread_id] = execution_graph
            result_state: dict[str, Any] = {}
            try:
                async for mode, chunk in execution_graph.astream(
                    {
                        "user_request": req.user_request,
                        "conversation_history": conversation_history,
                        "execution_thread_id": f"{thread_id}:execution:{uuid.uuid4()}",
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

            active_harness.sessions.update_session(thread_id, status="active")
            await active_harness._runtime.persist_session_to_memory(thread_id)
            active_harness.sessions.save_message(thread_id, sender="agent", text=output_text)
            active_harness._execution_graphs.pop(thread_id, None)
            done_ev = _json.dumps({
                "type": "done",
                "output": output_text,
                "status": result_state.get("status", "complete"),
            })
            yield f"data: {done_ev}\n\n"


        except Exception as exc:
            import traceback
            logger.error(f"Stream run error: {exc}\n{traceback.format_exc()}")
            active_harness._execution_graphs.pop(thread_id, None)
            if _is_groq_rate_limit(exc):
                rl_ev = _json.dumps(_make_rate_limit_event(chosen_model))
                yield f"data: {rl_ev}\n\n"
                done_msg = "🚫 Groq rate limit reached. Please switch to an Ollama model to continue."
                done_ev = _json.dumps({"type": "done", "output": done_msg, "status": "rate_limited"})
            else:
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
    if "messages" not in state or not state["messages"]:
        state["messages"] = messages
    return state


@app.post("/runs/{thread_id}/resume")
async def resume_run(thread_id: str, req: ResumeRequest) -> dict[str, Any]:
    global harness
    if harness is None:
        raise HTTPException(status_code=500, detail="Harness not initialized.")

    active_harness = harness
    result = await active_harness.resume(thread_id, approved=req.approved)
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
