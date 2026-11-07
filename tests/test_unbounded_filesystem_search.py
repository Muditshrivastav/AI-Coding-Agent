import asyncio
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends import filesystem as filesystem_backend
from deepagents.middleware import filesystem as filesystem_middleware

from agent.guardrail import HarnessGuard
from frameworks.deepagents_backend import GuardedShellBackend


class AllowReadGuard(HarnessGuard):
    def __init__(self) -> None:
        pass

    def enforce(self, *_: object, **__: object) -> None:
        return None


def _make_backend(root_dir: Path) -> GuardedShellBackend:
    return GuardedShellBackend(
        FilesystemBackend(root_dir=str(root_dir)),
        AllowReadGuard(),
    )


def test_guarded_filesystem_searches_have_no_deadline(tmp_path, monkeypatch):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "match.py").write_text("needle\n", encoding="utf-8")
    backend = _make_backend(tmp_path)
    monkeypatch.setattr(filesystem_backend, "_DEFAULT_GLOB_TIMEOUT", 0)
    monkeypatch.setattr(filesystem_backend, "DEFAULT_GREP_TIMEOUT", 0)

    glob_result = backend.glob("*.py", path="/")
    grep_result = backend.grep("needle", path="/")

    assert filesystem_middleware.GLOB_TIMEOUT is None
    assert [match["path"] for match in glob_result.matches] == ["/nested/match.py"]
    assert glob_result.truncated is False
    assert grep_result.matches == [
        {"path": "/nested/match.py", "line": 1, "text": "needle"}
    ]
    assert grep_result.truncated is False


def test_async_guarded_filesystem_searches_have_no_deadline(tmp_path, monkeypatch):
    (tmp_path / "match.txt").write_text("needle\n", encoding="utf-8")
    backend = _make_backend(tmp_path)
    monkeypatch.setattr(filesystem_backend, "_DEFAULT_GLOB_TIMEOUT", 0)
    monkeypatch.setattr(filesystem_backend, "DEFAULT_GREP_TIMEOUT", 0)

    async def exercise():
        glob_result = await backend.aglob("*.txt", path="/")
        grep_result = await backend.agrep("needle", path="/")
        return glob_result, grep_result

    glob_result, grep_result = asyncio.run(exercise())

    assert [match["path"] for match in glob_result.matches] == ["/match.txt"]
    assert grep_result.matches == [
        {"path": "/match.txt", "line": 1, "text": "needle"}
    ]
