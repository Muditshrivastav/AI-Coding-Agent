"""
frameworks/vscode_workspace.py — Resolves the VS Code workspace root directory.

Priority order:
1. AGENT_ROOT_DIR environment variable (explicit override).
2. VS Code's VSCODE_CWD env variable (set when running from VS Code terminal).
3. VS Code's multi-root workspace file (.code-workspace) — scans parent dirs.
4. The directory containing the nearest .git root — most repos are opened this way.
5. The process current working directory as a last resort.

Usage:
    from frameworks.vscode_workspace import resolve_workspace_root
    root = resolve_workspace_root()   # e.g. 'D:\\Projects\\my-app'
"""

import json
import os


def _find_upward(start: str, filenames: tuple[str, ...]) -> str | None:
    """Walk up the directory tree looking for any of the given filenames."""
    current = os.path.abspath(start)
    while True:
        for name in filenames:
            candidate = os.path.join(current, name)
            if os.path.exists(candidate):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _vscode_workspace_file(start: str) -> str | None:
    """Look for a .code-workspace file in start dir or any parent and return
    the first folder listed inside it (the primary workspace root)."""
    current = os.path.abspath(start)
    while True:
        for entry in os.scandir(current):
            if entry.name.endswith(".code-workspace") and entry.is_file():
                try:
                    with open(entry.path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    folders = data.get("folders", [])
                    if folders:
                        path = folders[0].get("path", "")
                        if os.path.isabs(path):
                            return path
                        return os.path.normpath(os.path.join(current, path))
                except Exception:
                    pass
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def resolve_workspace_root(default: str = ".") -> str:
    """Return the absolute path of the VS Code workspace root to use as root_dir.

    Detection order:
    1. ``AGENT_ROOT_DIR`` env var (explicit user override).
    2. ``VSCODE_CWD`` env var (set by VS Code terminal).
    3. `.code-workspace` file found in cwd or any ancestor directory.
    4. `.git` directory found in cwd or any ancestor directory.
    5. ``default`` (falls back to the process cwd when default=".").
    """
    # 1. Explicit override
    explicit = os.environ.get("AGENT_ROOT_DIR", "").strip()
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)

    # 2. VS Code terminal sets VSCODE_CWD to the workspace folder
    vscode_cwd = os.environ.get("VSCODE_CWD", "").strip()
    if vscode_cwd and os.path.isdir(vscode_cwd):
        return os.path.abspath(vscode_cwd)

    cwd = os.getcwd()

    # 3. .code-workspace file (multi-root workspace)
    workspace_file_root = _vscode_workspace_file(cwd)
    if workspace_file_root and os.path.isdir(workspace_file_root):
        return workspace_file_root

    # 4. Git root
    git_root = _find_upward(cwd, (".git",))
    if git_root:
        return git_root

    # 5. Fallback
    return os.path.abspath(default)
