"""
interfaces/acp_server.py - Agent Client Protocol (ACP) server for editor integration (e.g. VS Code).
"""

import asyncio
from agent.core import CodingAgentHarness


async def main() -> None:
    harness = CodingAgentHarness(root_dir=".", tools=[])
    print("[ACP Server] Coding Agent Harness initialized.")
    try:
        from acp import run_agent
        await run_agent(harness.agent, name="coding-agent-harness")
    except ImportError:
        print("[ACP Server] 'acp' / 'deepagents-acp' package not installed in environment. Ready for connection.")


if __name__ == "__main__":
    asyncio.run(main())
