"""Launch the API with an asyncio event loop compatible with Psycopg on Windows."""

import argparse
import asyncio
import selectors
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Coding Agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "interfaces.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="interfaces.run_api:selector_loop_factory",
    )


if __name__ == "__main__":
    main()
