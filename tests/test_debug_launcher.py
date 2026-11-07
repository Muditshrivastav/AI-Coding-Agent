import traceback
import selectors
import sys
from types import SimpleNamespace
from interfaces import run_api
import uvicorn

def test_debug():
    try:
        monkeypatch = None
        import pytest
    except Exception:
        pass
    # replicate test 3
    run_api.sys = SimpleNamespace(platform="win32")
    created_selectors = []
    select_selector = selectors.SelectSelector
    run_api.selectors = SimpleNamespace(
        SelectSelector=lambda: created_selectors.append(select_selector())
        or created_selectors[-1]
    )
    config = uvicorn.Config(
        "interfaces.api_server:app",
        loop="interfaces.run_api:selector_loop_factory",
    )
    loop_factory = config.get_loop_factory()
    with open("/debug.txt", "w") as f:
        f.write("loop_factory=%r\n" % (loop_factory,))
    loop = loop_factory()
    with open("/debug.txt", "a") as f:
        f.write("loop=%r\n" % (loop,))
        f.write("created=%d\n" % len(created_selectors))
    try:
        assert isinstance(loop, __import__("asyncio").SelectorEventLoop)
    finally:
        loop.close()
