"""
tools/github_oauth.py - GitHub OAuth 2.0 Authorization Code flow.

Flow
----
1.  The frontend (or Streamlit) sends the user to GET /auth/github?user_id=<id>.
    That endpoint calls ``build_authorization_url(user_id)`` and returns the URL.
2.  GitHub redirects back to GET /auth/github/callback?code=<code>&state=<user_id>.
    That endpoint calls ``exchange_code_for_token(code)`` and stores the result in
    the module-level ``vault`` singleton:  vault.set(user_id, "github", token).
3.  The build-agent (or any caller) resolves the token at runtime:
        token = vault.get(user_id)
    and passes it to ExternalToolsManager(github_token=token).

Required env vars (.env)
------------------------
    GITHUB_CLIENT_ID      - OAuth App Client ID
    GITHUB_CLIENT_SECRET  - OAuth App Client Secret
    GITHUB_REDIRECT_URI   - Callback URL registered in the OAuth App
                            (default: http://localhost:8000/auth/github/callback)

No Personal Access Token (PAT) is ever used, stored in config, or exposed in the UI.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# GitHub OAuth 2.0 endpoints
# ---------------------------------------------------------------------------
_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"

# Default scopes required by the coding agent
#   repo      - full repository access (read, write, push)
#   read:user - identify the authenticated user
_DEFAULT_SCOPES = "repo read:user"


# ---------------------------------------------------------------------------
# In-memory Token Vault
# ---------------------------------------------------------------------------

class TokenVault:
    """Lightweight in-memory token store keyed by (user_id, provider).

    Production note: swap the internal dict for an encrypted persistent
    store (Supabase, Redis, AWS Secrets Manager) — the public interface
    is identical.

    Usage::

        vault = TokenVault()
        vault.set("user-123", "github", "gho_...")
        token = vault.get("user-123")           # -> "gho_..." or None
        vault.revoke("user-123")
    """

    def __init__(self) -> None:
        # { user_id: { provider: token } }
        self._store: dict[str, dict[str, str]] = {}

    def set(self, user_id: str, provider: str, token: str) -> None:
        """Store (or overwrite) a token for a user+provider pair."""
        self._store.setdefault(user_id, {})[provider] = token

    def get(self, user_id: str, provider: str = "github") -> str | None:
        """Return the stored token, or None if the user has not yet authorised."""
        return self._store.get(user_id, {}).get(provider)

    def revoke(self, user_id: str, provider: str = "github") -> None:
        """Remove a token (e.g. on logout or token rotation)."""
        self._store.get(user_id, {}).pop(provider, None)

    def has(self, user_id: str, provider: str = "github") -> bool:
        """Return True if a live token exists for this user+provider."""
        return bool(self.get(user_id, provider))


# Module-level singleton — imported and shared by FastAPI routes in api_server.py
vault = TokenVault()


# ---------------------------------------------------------------------------
# OAuth step 1 — build the consent-screen URL
# ---------------------------------------------------------------------------

def build_authorization_url(
    user_id: str,
    scopes: str = _DEFAULT_SCOPES,
    client_id: str | None = None,
    redirect_uri: str | None = None,
) -> str:
    """Return the GitHub OAuth authorization URL to redirect the user to.

    The ``user_id`` is embedded as the ``state`` parameter so the callback
    can map code -> token back to the correct user without a server-side
    session cookie.

    Args:
        user_id:      Opaque identifier for the requesting user / session.
        scopes:       Space-separated GitHub OAuth scopes.
        client_id:    Override GITHUB_CLIENT_ID env var.
        redirect_uri: Override GITHUB_REDIRECT_URI env var.

    Raises:
        EnvironmentError: if GITHUB_CLIENT_ID is not configured.
    """
    cid = client_id or os.getenv("GITHUB_CLIENT_ID", "")
    redir = redirect_uri or os.getenv(
        "GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback"
    )
    if not cid:
        raise EnvironmentError(
            "GITHUB_CLIENT_ID is not set. "
            "Create a GitHub OAuth App and add the Client ID to your .env file."
        )

    params: dict[str, str] = {
        "client_id": cid,
        "redirect_uri": redir,
        "scope": scopes,
        "state": user_id,   # round-tripped by GitHub; used to identify the user
    }
    return f"{_GITHUB_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# OAuth step 2 — exchange the temporary code for an access token
# ---------------------------------------------------------------------------

async def exchange_code_for_token(
    code: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
) -> str:
    """POST to GitHub's token endpoint and return the access token string.

    Args:
        code:          The one-time code returned by GitHub in the callback URL.
        client_id:     Override GITHUB_CLIENT_ID env var.
        client_secret: Override GITHUB_CLIENT_SECRET env var.
        redirect_uri:  Override GITHUB_REDIRECT_URI env var.

    Returns:
        The raw GitHub access token (prefix ``gho_`` for OAuth tokens).

    Raises:
        httpx.HTTPStatusError: if GitHub returns a non-2xx HTTP status.
        ValueError:            if the response body contains an ``error`` field
                               (e.g. ``bad_verification_code``, expired code).
    """
    cid = client_id or os.getenv("GITHUB_CLIENT_ID", "")
    secret = client_secret or os.getenv("GITHUB_CLIENT_SECRET", "")
    redir = redirect_uri or os.getenv(
        "GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback"
    )

    async with httpx.AsyncClient(timeout=15.0) as http:
        response = await http.post(
            _GITHUB_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": cid,
                "client_secret": secret,
                "code": code,
                "redirect_uri": redir,
            },
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()

    if "error" in body:
        raise ValueError(
            f"GitHub OAuth error '{body.get('error')}': "
            f"{body.get('error_description', 'no description')}"
        )

    token: str | None = body.get("access_token")
    if not token:
        raise ValueError(f"No access_token in GitHub token response: {body}")

    return token
