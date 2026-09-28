"""
tools/api_tools.py - All real API-backed StructuredTools for the coding agent.

Integrates every service configured in .env and exposes each as a LangChain
StructuredTool the build-agent can call directly:

    • Tavily          — web search & documentation lookup
    • GitHub REST API — issues, PRs, repo info, commit history (uses GITHUB_PAT)
    • Supabase        — database queries via REST API
    • Render          — deployment status checks
    • Groq            — fast LLM inference for sub-tasks (summarise, classify)
    • LangSmith       — fetch run traces for self-debugging

All credentials are read from environment variables (.env). No keys are
hard-coded here.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)


# ===========================================================================
# Input schemas
# ===========================================================================

class _TavilyInput(BaseModel):
    query: str = Field(description="Search query to look up on the web.")
    max_results: int = Field(default=5, description="Number of results to return (1-10).")


class _GitHubIssueInput(BaseModel):
    owner: str = Field(description="GitHub repo owner (username or org).")
    repo: str = Field(description="GitHub repo name.")
    title: str = Field(description="Issue title.")
    body: str = Field(default="", description="Issue body / description.")
    labels: list[str] = Field(default_factory=list, description="Optional list of label names.")


class _GitHubListIssuesInput(BaseModel):
    owner: str = Field(description="GitHub repo owner.")
    repo: str = Field(description="GitHub repo name.")
    state: str = Field(default="open", description="Issue state: 'open', 'closed', or 'all'.")
    limit: int = Field(default=10, description="Max number of issues to return.")


class _GitHubPRInput(BaseModel):
    owner: str = Field(description="GitHub repo owner.")
    repo: str = Field(description="GitHub repo name.")
    title: str = Field(description="PR title.")
    body: str = Field(default="", description="PR description.")
    head: str = Field(description="Branch name with the changes.")
    base: str = Field(default="main", description="Branch to merge into.")


class _GitHubRepoInfoInput(BaseModel):
    owner: str = Field(description="GitHub repo owner.")
    repo: str = Field(description="GitHub repo name.")


class _GitHubCommitsInput(BaseModel):
    owner: str = Field(description="GitHub repo owner.")
    repo: str = Field(description="GitHub repo name.")
    branch: str = Field(default="main", description="Branch name.")
    limit: int = Field(default=10, description="Max number of commits to return.")


class _SupabaseQueryInput(BaseModel):
    table: str = Field(description="Supabase table name to query.")
    select: str = Field(default="*", description="Columns to select (default: all).")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs to filter rows by equality (e.g. {'status': 'active'}).",
    )
    limit: int = Field(default=20, description="Max rows to return.")


class _RenderStatusInput(BaseModel):
    service_id: str = Field(
        default="",
        description="Render service ID to check. If empty, lists all services.",
    )


class _GroqInferInput(BaseModel):
    prompt: str = Field(description="The prompt to send to Groq for fast inference.")
    model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model to use (e.g. 'llama-3.1-8b-instant', 'mixtral-8x7b-32768').",
    )
    max_tokens: int = Field(default=512, description="Max tokens for the response.")


class _LangSmithRunInput(BaseModel):
    run_name: str = Field(default="", description="Filter runs by name (partial match).")
    limit: int = Field(default=5, description="Number of recent runs to fetch.")


# ===========================================================================
# Tool implementations
# ===========================================================================

def _tavily_search(query: str, max_results: int = 5) -> str:
    """Web search via Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key", "")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in .env."
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": min(max(max_results, 1), 10),
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("answer", "")
        results = data.get("results", [])
        lines = []
        if answer:
            lines.append(f"[Answer] {answer}\n")
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'No title')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('content', '')[:300]}\n")
        return "\n".join(lines) if lines else "No results found."
    except Exception as exc:
        return f"Tavily search error: {exc}"


def _github_create_issue(
    owner: str, repo: str, title: str, body: str = "", labels: list[str] | None = None
) -> str:
    """Create a GitHub issue via REST API."""
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        return "Error: GITHUB_PAT is not set in .env."
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"title": title, "body": body, "labels": labels or []},
            )
            resp.raise_for_status()
            issue = resp.json()
        return (
            f"Issue created: #{issue['number']} — {issue['title']}\n"
            f"URL: {issue['html_url']}"
        )
    except Exception as exc:
        return f"GitHub create issue error: {exc}"


def _github_list_issues(owner: str, repo: str, state: str = "open", limit: int = 10) -> str:
    """List GitHub issues for a repo."""
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        return "Error: GITHUB_PAT is not set in .env."
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                params={"state": state, "per_page": min(limit, 50)},
            )
            resp.raise_for_status()
            issues = resp.json()
        if not issues:
            return f"No {state} issues found in {owner}/{repo}."
        lines = [f"[{state.upper()} ISSUES — {owner}/{repo}]"]
        for issue in issues[:limit]:
            lines.append(f"  #{issue['number']} {issue['title']}  ({issue['html_url']})")
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub list issues error: {exc}"


def _github_create_pr(
    owner: str, repo: str, title: str, head: str, base: str = "main", body: str = ""
) -> str:
    """Create a GitHub Pull Request via REST API."""
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        return "Error: GITHUB_PAT is not set in .env."
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                f"https://api.github.com/repos/{owner}/{repo}/pulls",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"title": title, "body": body, "head": head, "base": base},
            )
            resp.raise_for_status()
            pr = resp.json()
        return (
            f"PR created: #{pr['number']} — {pr['title']}\n"
            f"URL: {pr['html_url']}\n"
            f"Status: {pr['state']}"
        )
    except Exception as exc:
        return f"GitHub create PR error: {exc}"


def _github_repo_info(owner: str, repo: str) -> str:
    """Fetch metadata about a GitHub repo."""
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"https://api.github.com/repos/{owner}/{repo}", headers=headers
            )
            resp.raise_for_status()
            r = resp.json()
        return (
            f"Repo: {r['full_name']}\n"
            f"Description: {r.get('description', 'N/A')}\n"
            f"Stars: {r['stargazers_count']}  Forks: {r['forks_count']}  Open issues: {r['open_issues_count']}\n"
            f"Default branch: {r['default_branch']}\n"
            f"Language: {r.get('language', 'N/A')}\n"
            f"URL: {r['html_url']}"
        )
    except Exception as exc:
        return f"GitHub repo info error: {exc}"


def _github_list_commits(owner: str, repo: str, branch: str = "main", limit: int = 10) -> str:
    """List recent commits on a branch."""
    token = os.getenv("GITHUB_PAT") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits",
                headers=headers,
                params={"sha": branch, "per_page": min(limit, 50)},
            )
            resp.raise_for_status()
            commits = resp.json()
        if not commits:
            return f"No commits found on branch '{branch}'."
        lines = [f"[COMMITS — {owner}/{repo}@{branch}]"]
        for c in commits[:limit]:
            sha = c["sha"][:7]
            msg = c["commit"]["message"].split("\n")[0]
            author = c["commit"]["author"]["name"]
            date = c["commit"]["author"]["date"][:10]
            lines.append(f"  {sha}  {date}  {author}: {msg}")
        return "\n".join(lines)
    except Exception as exc:
        return f"GitHub list commits error: {exc}"


def _supabase_query(
    table: str, select: str = "*", filters: dict[str, Any] | None = None, limit: int = 20
) -> str:
    """Query a Supabase table via REST API."""
    url = os.getenv("supabase_url") or os.getenv("SUPABASE_URL", "")
    anon_key = os.getenv("supabase_anon_key") or os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not anon_key:
        return "Error: supabase_url or supabase_anon_key is not set in .env."
    try:
        endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
        params: dict[str, Any] = {"select": select, "limit": limit}
        if filters:
            for key, val in filters.items():
                params[f"{key}"] = f"eq.{val}"

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                endpoint,
                headers={
                    "apikey": anon_key,
                    "Authorization": f"Bearer {anon_key}",
                    "Content-Type": "application/json",
                },
                params=params,
            )
            resp.raise_for_status()
            rows = resp.json()

        if not rows:
            return f"No rows returned from table '{table}'."
        # Render as a compact table
        if isinstance(rows, list) and rows:
            keys = list(rows[0].keys())
            header = " | ".join(keys)
            sep = "-" * len(header)
            lines = [f"[{table}]", header, sep]
            for row in rows:
                lines.append(" | ".join(str(row.get(k, "")) for k in keys))
            return "\n".join(lines)
        return str(rows)
    except Exception as exc:
        return f"Supabase query error: {exc}"


def _render_status(service_id: str = "") -> str:
    """Check Render service deployment status."""
    api_key = os.getenv("RENDER_API_KEY") or os.getenv("render_api_key", "")
    if not api_key:
        return "Error: RENDER_API_KEY is not set in .env."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            if service_id:
                resp = client.get(
                    f"https://api.render.com/v1/services/{service_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                svc = resp.json()
                return (
                    f"Service: {svc.get('name', 'N/A')} (ID: {svc.get('id', 'N/A')})\n"
                    f"Type: {svc.get('type', 'N/A')}\n"
                    f"Status: {svc.get('suspended', 'active')}\n"
                    f"URL: {svc.get('serviceDetails', {}).get('url', 'N/A')}"
                )
            else:
                resp = client.get(
                    "https://api.render.com/v1/services",
                    headers=headers,
                    params={"limit": 10},
                )
                resp.raise_for_status()
                services = resp.json()
                if not services:
                    return "No Render services found."
                lines = ["[RENDER SERVICES]"]
                for item in services:
                    svc = item.get("service", item)
                    lines.append(
                        f"  {svc.get('name', 'N/A')} — ID: {svc.get('id', 'N/A')} — {svc.get('type', 'N/A')}"
                    )
                return "\n".join(lines)
    except Exception as exc:
        return f"Render status error: {exc}"


def _groq_infer(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 512) -> str:
    """Run fast inference on Groq for sub-tasks like summarisation or classification."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "Error: GROQ_API_KEY is not set in .env."
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"Groq inference error: {exc}"


def _langsmith_recent_runs(run_name: str = "", limit: int = 5) -> str:
    """Fetch recent LangSmith trace runs for self-debugging."""
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "agentic-coder")
    if not api_key:
        return "Error: LANGSMITH_API_KEY is not set in .env."
    try:
        params: dict[str, Any] = {
            "project_name": project,
            "limit": limit,
            "is_root": "true",
        }
        if run_name:
            params["name"] = run_name

        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://api.smith.langchain.com/runs",
                headers={"x-api-key": api_key},
                params=params,
            )
            resp.raise_for_status()
            runs = resp.json()

        if not runs:
            return f"No runs found in project '{project}'."
        lines = [f"[LANGSMITH RUNS — {project}]"]
        for run in runs[:limit]:
            name = run.get("name", "N/A")
            status = run.get("status", "N/A")
            start = (run.get("start_time") or "")[:19]
            error = run.get("error") or ""
            lines.append(f"  [{status.upper()}] {start}  {name}" + (f"  ERR: {error[:80]}" if error else ""))
        return "\n".join(lines)
    except Exception as exc:
        return f"LangSmith runs error: {exc}"


# ===========================================================================
# APIToolset — single source of all tools
# ===========================================================================

class APIToolset:
    """
    Consolidated API toolset for the coding agent.

    Exposes every real API service configured in .env as a LangChain StructuredTool:
      - tavily_web_search       → Tavily web & doc search
      - github_create_issue     → Create a GitHub issue
      - github_list_issues      → List open/closed issues
      - github_create_pr        → Open a Pull Request
      - github_repo_info        → Repo metadata (stars, language, default branch)
      - github_list_commits     → Recent commits on a branch
      - supabase_query          → Query Supabase tables via REST
      - render_service_status   → Check Render deployment status
      - groq_infer              → Fast Groq LLM inference for sub-tasks
      - langsmith_recent_runs   → Fetch recent trace runs for self-debugging
    """

    def __init__(
        self,
        base_url: str = "",           # kept for backwards compatibility
        auth_token: str | None = None,  # kept for backwards compatibility
    ) -> None:
        self._base_url = base_url
        self._auth_token = auth_token

    def get_tools(self) -> list[StructuredTool]:
        """Return all real API-backed StructuredTools."""
        return [
            # ── Web Search ──────────────────────────────────────────────────
            StructuredTool.from_function(
                func=_tavily_search,
                name="tavily_web_search",
                description=(
                    "Search the web or look up documentation using Tavily. "
                    "Use for library docs, error messages, APIs, and current events."
                ),
                args_schema=_TavilyInput,
            ),

            # ── GitHub REST API ─────────────────────────────────────────────
            StructuredTool.from_function(
                func=_github_create_issue,
                name="github_create_issue",
                description=(
                    "Create a GitHub issue in a repository. "
                    "Use to track bugs, feature requests, or tasks. Requires GITHUB_PAT."
                ),
                args_schema=_GitHubIssueInput,
            ),
            StructuredTool.from_function(
                func=_github_list_issues,
                name="github_list_issues",
                description=(
                    "List open or closed GitHub issues for a repository. "
                    "Useful for checking existing tasks or bugs before creating duplicates."
                ),
                args_schema=_GitHubListIssuesInput,
            ),
            StructuredTool.from_function(
                func=_github_create_pr,
                name="github_create_pr",
                description=(
                    "Open a Pull Request on GitHub from a feature branch into a base branch. "
                    "Requires GITHUB_PAT with repo scope."
                ),
                args_schema=_GitHubPRInput,
            ),
            StructuredTool.from_function(
                func=_github_repo_info,
                name="github_repo_info",
                description=(
                    "Fetch metadata about a GitHub repository: description, stars, forks, "
                    "default branch, and primary language."
                ),
                args_schema=_GitHubRepoInfoInput,
            ),
            StructuredTool.from_function(
                func=_github_list_commits,
                name="github_list_commits",
                description=(
                    "List recent commits on a GitHub branch. "
                    "Use to check what changed recently before making edits."
                ),
                args_schema=_GitHubCommitsInput,
            ),

            # ── Supabase ────────────────────────────────────────────────────
            StructuredTool.from_function(
                func=_supabase_query,
                name="supabase_query",
                description=(
                    "Query a Supabase table via the REST API. "
                    "Use to read application data, user records, or session state."
                ),
                args_schema=_SupabaseQueryInput,
            ),

            # ── Render ──────────────────────────────────────────────────────
            StructuredTool.from_function(
                func=_render_status,
                name="render_service_status",
                description=(
                    "Check the status of a Render deployment service. "
                    "If service_id is empty, lists all services. Requires RENDER_API_KEY."
                ),
                args_schema=_RenderStatusInput,
            ),

            # ── Groq Inference ──────────────────────────────────────────────
            StructuredTool.from_function(
                func=_groq_infer,
                name="groq_infer",
                description=(
                    "Run a prompt through Groq for fast, cheap LLM inference. "
                    "Use for summarisation, classification, or any sub-task that doesn't need the full agent."
                ),
                args_schema=_GroqInferInput,
            ),

            # ── LangSmith ───────────────────────────────────────────────────
            StructuredTool.from_function(
                func=_langsmith_recent_runs,
                name="langsmith_recent_runs",
                description=(
                    "Fetch recent LangSmith trace runs for the project. "
                    "Use to inspect failed runs, latency, or errors for self-debugging."
                ),
                args_schema=_LangSmithRunInput,
            ),
        ]
