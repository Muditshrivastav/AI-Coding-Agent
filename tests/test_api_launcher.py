import asyncio
import selectors
import sys
from types import SimpleNamespace

from interfaces import run_api


def test_selector_loop_factory_uses_windows_selector_loop(monkeypatch):
    monkeypatch.setattr(run_api, "sys", SimpleNamespace(platform="win32"))
    created_selectors = []
    expected_loop = object()

    def create_loop(selector):
        created_selectors.append(selector)
        return expected_loop

    monkeypatch.setattr(
        run_api,
        "asyncio",
        SimpleNamespace(SelectorEventLoop=create_loop),
    )

    loop = run_api.selector_loop_factory()

    assert loop is expected_loop
    assert len(created_selectors) == 1
    assert isinstance(created_selectors[0], selectors.SelectSelector)


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
    """Verify uvicorn resolves the loop string and the factory uses SelectSelector."""
    import uvicorn

    # Patch sys and asyncio on run_api so no real event loop is ever created.
    expected_loop = object()
    created_selectors: list[selectors.SelectSelector] = []
    real_SelectSelector = selectors.SelectSelector

    def fake_SelectorEventLoop(selector):
        created_selectors.append(selector)
        return expected_loop

    monkeypatch.setattr(run_api, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        run_api,
        "asyncio",
        SimpleNamespace(SelectorEventLoop=fake_SelectorEventLoop),
    )
    monkeypatch.setattr(
        run_api,
        "selectors",
        SimpleNamespace(SelectSelector=real_SelectSelector),
    )

    config = uvicorn.Config(
        "interfaces.api_server:app",
        loop="interfaces.run_api:selector_loop_factory",
    )

    loop_factory = config.get_loop_factory()
    assert loop_factory is not None, "uvicorn did not resolve the loop factory"

    loop = loop_factory()

    assert loop is expected_loop
    assert len(created_selectors) == 1
    assert isinstance(created_selectors[0], selectors.SelectSelector)