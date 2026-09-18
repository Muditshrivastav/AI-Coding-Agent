"""
nodes/design_subagent.py - Design subagent definition.
Reads PLAN.md dynamically and writes ARCHITECTURE.md per run.
Connects to the draw.io hosted MCP server so the agent can create and
export architecture diagrams directly via draw.io tools.
"""

from typing import Any

import os

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langgraph.types import interrupt

from agent.guardrail import HarnessGuard
from frameworks.mcp_client import DrawioMCPClient, drawio_mcp_client

DESIGN_SYSTEM_PROMPT = """You are the design-agent for an autonomous coding harness.
Your task is to read PLAN.md from the root directory and produce ARCHITECTURE.md.

Follow these strict requirements:
1. Read PLAN.md via read_file before producing any output.
2. Outline components, inter-module communication, data flow, tech choices, and file boundaries.
3. Structure the output into ARCHITECTURE.md so the build subagent can decompose it into concrete todos.
4. Use the available draw.io tools to create a visual architecture diagram and export it alongside ARCHITECTURE.md.
"""


class _WriteArchArgs(BaseModel):
    content: str = Field(description="The full Markdown content to write to ARCHITECTURE.md.")


async def make_design_subagent(
    root_dir: str = ".",
) -> tuple[Any, DrawioMCPClient]:
    """Build a deep design agent wired with draw.io MCP tools and FileSystemBackend.

    What happens inside:
      1. ``ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)`` is used as the LLM.
      2. ``FilesystemBackend(root_dir)`` lets the agent read files from disk.
      3. A HarnessGuard-protected ``write_architecture`` tool gates every write to
         ARCHITECTURE.md through permissions.json before touching disk.
      4. ``DrawioMCPClient`` POSTs the MCP initialize request to https://mcp.draw.io/mcp,
         extracts ``Mcp-Session-Id``, and returns draw.io LangChain tools.
      5. All of the above are forwarded to ``create_deep_agent``.

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
    guard = HarnessGuard(os.path.join(root_dir, "harness", "permissions.json"))
    arch_path = os.path.join(root_dir, "ARCHITECTURE.md")

    def _write_architecture(content: str) -> str:
        """Guard-enforced write of ARCHITECTURE.md."""
        guard.enforce(
            "Edit(ARCHITECTURE.md)",
            interrupt_fn=interrupt,
            description="Design agent requests write to ARCHITECTURE.md",
        )
        os.makedirs(os.path.dirname(os.path.abspath(arch_path)), exist_ok=True)
        with open(arch_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"ARCHITECTURE.md written successfully ({len(content)} chars)."

    write_arch_tool = StructuredTool.from_function(
        func=_write_architecture,
        name="write_architecture",
        description=(
            "Write the complete architecture document to ARCHITECTURE.md. "
            "Use this tool (not a raw file-write) to ensure HarnessGuard approves the write. "
            "Provide the full Markdown content as the 'content' argument."
        ),
        args_schema=_WriteArchArgs,
    )

    llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.2)
    backend = FilesystemBackend(root_dir=root_dir)

    drawio: DrawioMCPClient = drawio_mcp_client()
    drawio_tools = await drawio.get_tools()  # POSTs initialize, then list_tools

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt=DESIGN_SYSTEM_PROMPT,
        tools=[write_arch_tool, *drawio_tools],
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
