"""
nodes/design_subagent.py - Design subagent definition.
Reads PLAN.md dynamically and writes ARCHITECTURE.md per run.
Connects to the draw.io hosted MCP server so the agent can create and
export architecture diagrams directly via draw.io tools.
"""

from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FileSystemBackend
from langchain_ollama import ChatOllama

from frameworks.mcp_client import DrawioMCPClient, drawio_mcp_client

DESIGN_SYSTEM_PROMPT = """You are the design-agent for an autonomous coding harness.
Your task is to read PLAN.md from the root directory and produce ARCHITECTURE.md.

Follow these strict requirements:
1. Read PLAN.md via read_file before producing any output.
2. Outline components, inter-module communication, data flow, tech choices, and file boundaries.
3. Structure the output into ARCHITECTURE.md so the build subagent can decompose it into concrete todos.
4. Use the available draw.io tools to create a visual architecture diagram and export it alongside ARCHITECTURE.md.
"""


async def make_design_subagent(
    root_dir: str = ".",
) -> tuple[Any, DrawioMCPClient]:
    """Build a deep design agent wired with draw.io MCP tools and FileSystemBackend.

    What happens inside:
      1. ``ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)`` is used as the LLM.
      2. ``FileSystemBackend(root_dir)`` lets the agent write ARCHITECTURE.md to disk.
      3. ``DrawioMCPClient`` POSTs the MCP initialize request to https://mcp.draw.io/mcp,
         extracts ``Mcp-Session-Id``, and returns draw.io LangChain tools.
      4. All of the above are forwarded to ``create_deep_agent``.

    Args:
        root_dir: Workspace root where ARCHITECTURE.md will be written.

    Returns:
        ``(deep_agent, drawio_client)`` — call ``await drawio_client.close()``
        after the agent run to tear down the MCP session and httpx client.

    Usage::

        agent, drawio = await make_design_subagent()
        try:
            await agent.ainvoke({...})
        finally:
            await drawio.close()
    """
    llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)
    backend = FileSystemBackend(root_dir=root_dir)

    drawio: DrawioMCPClient = drawio_mcp_client()
    drawio_tools = await drawio.get_tools()  # POSTs initialize, then list_tools

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=DESIGN_SYSTEM_PROMPT,
        tools=drawio_tools,
    )
    return agent, drawio


# ---------------------------------------------------------------------------
# Convenience static spec (no draw.io tools) for declarative harnesses that
# do not await an async factory.
# ---------------------------------------------------------------------------
design_subagent: dict[str, Any] = {
    "name": "design-agent",
    "description": "Use AFTER planning-agent. Reads PLAN.md and writes ARCHITECTURE.md dynamically.",
    "system_prompt": DESIGN_SYSTEM_PROMPT,
    "factory": make_design_subagent,
}
