"""
interfaces/acp_server.py - Agent Client Protocol (ACP) server for VS Code editor integration.

When launched from the VS Code integrated terminal or extension, this server
automatically detects the open workspace root (via VSCODE_CWD / .git root / .code-workspace)
and wires the coding agent to write code directly in that project folder.

UI Integration:
- If the backend API server is running on http://localhost:8000, runs automatically sync
  with the Web Frontend (http://localhost:5173) in real-time.
- If running standalone, runs locally in the workspace with direct host shell access.

Run:
    python interfaces/acp_server.py
    # or
    python -m interfaces.acp_server
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request
import uuid

# Ensure repository root is on sys.path even when executed directly from VS Code
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frameworks.vscode_workspace import resolve_workspace_root

logging.basicConfig(
    level=logging.INFO,
    format="[ACP] %(levelname)s %(message)s",
)
logger = logging.getLogger("acp_server")

API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8000")


def extract_response_text(run_output: dict[str, Any]) -> str:
    """Extract human-readable text from harness.run() output."""
    if run_output.get("status") == "blocked":
        return f"⚠️ Request blocked: {run_output.get('reason', 'Security policy violation')}"
    if run_output.get("status") == "awaiting_approval":
        return f"⏸️ Awaiting approval: {run_output.get('interrupt', '')}"
    res = run_output.get("result", {})
    if isinstance(res, dict):
        messages = res.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return str(last.content)
            elif isinstance(last, dict) and "content" in last:
                return str(last["content"])
    return str(res)


async def execute_task_with_ui_sync(
    prompt: str,
    thread_id: str,
    harness: Any,
) -> tuple[dict[str, Any], bool]:
    """Execute run through the API server if active (so UI updates live),
    otherwise execute locally via harness."""
    loop = asyncio.get_running_loop()

    def _call_api() -> dict[str, Any] | None:
        try:
            # 1. Quick health check to see if backend API is reachable
            check_req = urllib.request.Request(
                f"{API_SERVER_URL}/health",
                headers={"User-Agent": "acp-client"},
            )
            with urllib.request.urlopen(check_req, timeout=1.0) as resp:
                if resp.status != 200:
                    return None

            # 2. Ensure session is registered on the API server so it appears in the UI
            sess_payload = json.dumps({
                "title": f"VS Code: {prompt[:28]}",
                "session_id": thread_id,
            }).encode("utf-8")
            s_req = urllib.request.Request(
                f"{API_SERVER_URL}/sessions",
                data=sess_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(s_req, timeout=2.0) as _:
                    pass
            except Exception:
                pass

            # 3. Post run to API server
            run_payload = json.dumps({
                "user_request": prompt,
                "thread_id": thread_id,
            }).encode("utf-8")
            r_req = urllib.request.Request(
                f"{API_SERVER_URL}/runs",
                data=run_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(r_req, timeout=300.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    api_result = await loop.run_in_executor(None, _call_api)
    if api_result is not None:
        return api_result, True

    # Fallback to local harness
    local_res = await harness.run(user_request=prompt, thread_id=thread_id)
    return local_res, False


async def run_stdio_jsonrpc(harness: Any, root_dir: str) -> None:
    """Built-in JSON-RPC 2.0 stdio transport compatible with ACP and IDE clients."""
    logger.info("📡 ACP stdio transport started. Listening for requests...")
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8").strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        response: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}

        if method == "initialize":
            response["result"] = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "coding-agent-harness",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "workspace": {"rootDir": root_dir, "rootUri": Path(root_dir).as_uri()},
                    "tools": {},
                    "prompts": {},
                },
            }
        elif method in ("session/new", "session/initialize"):
            thread_id = params.get("threadId") or params.get("sessionId") or str(uuid.uuid4())
            response["result"] = {"sessionId": thread_id, "root_dir": root_dir}
        elif method in ("session/prompt", "agent/prompt", "chat", "message"):
            prompt = (
                params.get("prompt")
                or params.get("text")
                or params.get("content")
                or params.get("message")
                or ""
            )
            thread_id = (
                params.get("threadId")
                or params.get("sessionId")
                or params.get("thread_id")
                or str(uuid.uuid4())
            )
            try:
                run_res, synced = await execute_task_with_ui_sync(prompt, thread_id, harness)
                response["result"] = {
                    "sessionId": thread_id,
                    "status": run_res.get("status", "complete"),
                    "content": extract_response_text(run_res),
                    "uiSynced": synced,
                }
            except Exception as e:
                response["error"] = {"code": -32603, "message": str(e)}
        elif method == "ping":
            response["result"] = "pong"
        else:
            response["error"] = {"code": -32601, "message": f"Method not found: {method}"}

        if msg_id is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


async def run_interactive_terminal(harness: Any, root_dir: str) -> None:
    """Interactive CLI loop when developer executes acp_server directly in VS Code terminal."""
    # Check if UI API is connected
    api_online = False
    try:
        req = urllib.request.Request(f"{API_SERVER_URL}/health", headers={"User-Agent": "acp-client"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            api_online = (resp.status == 200)
    except Exception:
        api_online = False

    print("\n" + "=" * 66)
    print("🤖  Coding Agent Harness — VS Code Direct Workspace Mode")
    print(f"📂  Target Root: {root_dir}")
    print("⚡  Sandbox: Local Host (writes code directly into files)")
    if api_online:
        print("🔗  UI Link: Connected to http://localhost:8000 (Live sync active)")
        print("🖥️  Frontend: Open http://localhost:5173 to watch runs live")
    else:
        print("ℹ️   UI Link: Offline (running standalone in terminal)")
        print("💡  To sync with Web UI: run 'uvicorn interfaces.api_server:app --port 8000'")
    print("==================================================================\n")
    print("Type your task below (or type 'exit' or 'quit' to stop):\n")

    thread_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()

    while True:
        try:
            prompt = await loop.run_in_executor(None, input, "agent> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Session ended.")
            break

        print(f"\n⏳ Working on: '{prompt}'...")
        try:
            res, synced = await execute_task_with_ui_sync(prompt, thread_id, harness)
            output = extract_response_text(res)
            print(f"\n✅ Result:\n{output}\n")
            if synced:
                print("✨ [Synced with Web UI at http://localhost:5173]\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


async def main() -> None:
    # Resolve the VS Code workspace root so the agent writes code into
    # whichever project folder is currently open in the editor.
    root_dir = resolve_workspace_root()
    logger.info(f"🗂️  Workspace root detected: {root_dir}")

    from agent.core import CodingAgentHarness

    harness = CodingAgentHarness(
        root_dir=root_dir,
        tools=[],
        # Run directly on the host shell — writes code straight to the workspace.
        sandbox_mode="local",
    )
    logger.info("✅ CodingAgentHarness initialized (sandbox_mode=local).")
    logger.info(f"   Target Directory: {root_dir}")

    # Check for official agent-client-protocol / acp package first
    acp_started = False
    for pkg in ("agent_client_protocol", "acp"):
        try:
            mod = __import__(pkg)
            if hasattr(mod, "run_agent"):
                logger.info(f"🚀 Starting ACP server via '{pkg}' package...")
                await mod.run_agent(harness.agent, name="coding-agent-harness")
                acp_started = True
                break
        except ImportError:
            continue
        except Exception as exc:
            logger.warning(f"Failed to start with {pkg}: {exc}")

    if not acp_started:
        if sys.stdin.isatty():
            await run_interactive_terminal(harness, root_dir)
        else:
            await run_stdio_jsonrpc(harness, root_dir)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
