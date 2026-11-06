import asyncio
import selectors
import sys
from types import SimpleNamespace

from interfaces import run_api


def test_selector_loop_factory_uses_windows_selector_loop(monkeypatch):
    monkeypatch.setattr(run_api, "sys", SimpleNamespace(platform="win32"))
    # Root cause: run_api.sys is the actual sys module. 
    # monkeypatch.setattr(run_api.sys, "platform", "win32") fails because sys.platform is read-only.
    # We must patch the 'sys' reference within the run_api module.

    loop = run_api.selector_loop_factory()

    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert isinstance(loop._selector, selectors.SelectSelector)
    finally:
        loop.close()


def test_main_passes_custom_loop_factory_to_uvicorn(monkeypatch):
    call = {}
    monkeypatch.setattr(sys, "argv", ["run_api"])
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: call.update(kwargs)),
    )

    run_api.main()

    assert call["loop"] == "interfaces.run_api:selector_loop_factory"


def test_uvicorn_can_load_and_use_loop_setup(monkeypatch):
    import uvicorn

    # Root cause: same as above, sys.platform is read-only.
    monkeypatch.setattr(run_api, "sys", SimpleNamespace(platform="win32"))
    config = uvicorn.Config(
        "interfaces.api_server:app",
        loop="interfaces.run_api:selector_loop_factory",
    )

    loop = config.get_loop_factory()()

    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert isinstance(loop._selector, selectors.SelectSelector)
    finally:
        loop.close()