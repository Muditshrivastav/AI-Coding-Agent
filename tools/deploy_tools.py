"""
tools/deploy_tools.py - API-based and CLI deployment triggers for Render and Vercel.
"""

import os
import subprocess
from typing import Any
from dotenv import load_dotenv
import httpx
from langchain_core.tools import StructuredTool

load_dotenv()


RENDER_MCP_URL = "https://mcp.render.com/mcp"
RENDER_API_BASE_URL = "https://api.render.com/v1"


class DeployToolset:
    """Manages deployment triggers executed via Render MCP (with REST API fallback) and Vercel."""

    def __init__(
        self,
        root_dir: str = ".",
        render_api_key: str | None = None,
        vercel_token: str | None = None,
        render_mcp_url: str = RENDER_MCP_URL,
    ) -> None:
        self._root_dir = root_dir
        self._render_api_key = render_api_key or os.getenv("RENDER_API_KEY", "")
        self._vercel_token = vercel_token or os.getenv("VERCEL_TOKEN", "")
        self._render_mcp_url = render_mcp_url

    def _render_mcp_deploy(self, service_id: str, clear_cache: bool = False) -> dict[str, Any]:
        """Primary: Attempts to trigger deployment via hosted Render MCP endpoint (https://mcp.render.com/mcp)."""
        api_key = self._render_api_key or os.getenv("RENDER_API_KEY", "")
        if not api_key:
            return {"success": False, "error": "RENDER_API_KEY is not set."}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                # 1. MCP initialize handshake
                init_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "ai-coding-agent", "version": "1.0.0"},
                    },
                }
                init_res = client.post(self._render_mcp_url, headers=headers, json=init_payload)
                if init_res.status_code not in (200, 201):
                    return {"success": False, "error": f"MCP init failed with status {init_res.status_code}: {init_res.text}"}

                # Capture session ID header if provided
                session_id = init_res.headers.get("Mcp-Session-Id")
                session_headers = dict(headers)
                if session_id:
                    session_headers["Mcp-Session-Id"] = session_id

                # 2. Notify initialized
                notify_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
                client.post(self._render_mcp_url, headers=session_headers, json=notify_payload)

                # 3. List available MCP tools to find deploy tool
                list_payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                }
                list_res = client.post(self._render_mcp_url, headers=session_headers, json=list_payload)
                deploy_tool_name = "create_deploy"
                if list_res.status_code in (200, 201):
                    tools_list = list_res.json().get("result", {}).get("tools", [])
                    for t in tools_list:
                        name = t.get("name", "")
                        if "deploy" in name.lower():
                            deploy_tool_name = name
                            break

                # 4. Invoke deployment tool via MCP
                call_payload = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": deploy_tool_name,
                        "arguments": {
                            "serviceId": service_id,
                            "clearCache": "clear" if clear_cache else "do_not_clear",
                        },
                    },
                }
                call_res = client.post(self._render_mcp_url, headers=session_headers, json=call_payload)
                if call_res.status_code in (200, 201):
                    data = call_res.json()
                    if "error" in data:
                        return {"success": False, "error": f"MCP tool call error: {data['error']}"}
                    result = data.get("result", {})
                    return {
                        "success": True,
                        "message": f"Render MCP deployment succeeded via {self._render_mcp_url} [{deploy_tool_name}]: {result}",
                    }

                return {"success": False, "error": f"MCP tool call HTTP {call_res.status_code}: {call_res.text}"}

        except Exception as exc:
            return {"success": False, "error": f"Render MCP connection error: {str(exc)}"}

    def _render_rest_deploy_fallback(self, service_id: str, clear_cache: bool = False) -> str:
        """Fallback: Trigger a deploy on Render via plain Render REST API."""
        api_key = self._render_api_key or os.getenv("RENDER_API_KEY", "")
        if not api_key:
            return "Render fallback deployment failed: RENDER_API_KEY is not set."
        if not service_id:
            return "Render fallback deployment failed: service_id is required."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "clearCache": "clear" if clear_cache else "do_not_clear"
        }
        url = f"{RENDER_API_BASE_URL}/services/{service_id}/deploys"

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code in (200, 201):
                    data = res.json()
                    deploy_id = data.get("id", "unknown")
                    status = data.get("status", "created")
                    return f"[REST API Fallback] Render deployment triggered successfully! Deploy ID: {deploy_id}, Status: {status}"
                return f"[REST API Fallback] Render deployment failed [{res.status_code}]: {res.text}"
        except Exception as e:
            return f"[REST API Fallback] Render deployment error: {str(e)}"

    def _render_deploy(self, service_id: str, clear_cache: bool = False) -> str:
        """Deploy to Render: uses hosted MCP endpoint https://mcp.render.com/mcp first, falling back to REST API."""
        if not service_id:
            return "Render deployment failed: service_id is required."

        # 1. Primary: hosted Render MCP endpoint
        mcp_res = self._render_mcp_deploy(service_id=service_id, clear_cache=clear_cache)
        if mcp_res.get("success"):
            return str(mcp_res.get("message"))

        # 2. Fallback: plain Render REST API call
        error_info = mcp_res.get("error", "Hosted MCP endpoint unavailable")
        fallback_msg = self._render_rest_deploy_fallback(service_id=service_id, clear_cache=clear_cache)
        return (
            f"[Render MCP Notification: {error_info}]\n"
            f"[Executing Fallback via Render REST API]:\n{fallback_msg}"
        )

    def _vercel_deploy(self, project_name: str = "", project_dir: str = "", prod: bool = False) -> str:
        """Deploy to Vercel via Vercel REST API or CLI."""
        token = self._vercel_token or os.getenv("VERCEL_TOKEN", "")

        # If project_dir is provided or token is not configured, fall back to CLI
        if project_dir or not token:
            flags = ["--prod"] if prod else []
            cmd = ["vercel", "deploy", *flags, "--yes"]
            try:
                res = subprocess.run(
                    cmd,
                    cwd=project_dir or self._root_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return res.stdout if res.returncode == 0 else f"Deploy failed: {res.stderr}"
            except Exception as e:
                return f"Deployment error: {str(e)}"

        # Trigger via Vercel REST API
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "name": project_name or "project",
            "target": "production" if prod else "preview",
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post("https://api.vercel.com/v13/deployments", headers=headers, json=payload)
                if res.status_code in (200, 201):
                    data = res.json()
                    url = data.get("url", "unknown")
                    return f"Vercel deployment triggered successfully! URL: {url}"
                return f"Vercel deployment failed [{res.status_code}]: {res.text}"
        except Exception as e:
            return f"Vercel deployment error: {str(e)}"

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self._render_deploy,
                name="render_deploy",
                description="Triggers deployment via hosted Render MCP endpoint (https://mcp.render.com/mcp) with automatic REST API fallback. Requires service_id. Requires HITL approval.",
            ),
            StructuredTool.from_function(
                func=self._vercel_deploy,
                name="vercel_deploy",
                description="Triggers a deployment via Vercel REST API or CLI. Requires HITL approval.",
            ),
        ]
