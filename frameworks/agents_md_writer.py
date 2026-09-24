"""
frameworks/agents_md_writer.py - Automatic AGENTS.md generator.

When the user submits a query (e.g. "build an OAuth feature for my login page"),
this module uses an LLM to synthesize a tailored AGENTS.md that describes:
  - What feature is being built
  - Which subagents are responsible for which phases
  - Specific conventions, forbidden actions, and acceptance criteria for this task

The generated AGENTS.md is written to harness/AGENTS.md (scoped to the
current session root_dir) BEFORE the main agent graph executes, so that
all subagents (planning, design, build) receive context tightly coupled
to the user actual request.

Design
------
- Uses a lightweight LLM call (separate from the main agent loop) to keep it
  fast and non-blocking.
- Falls back gracefully (preserving the existing AGENTS.md) if generation fails.
- Writes are idempotent: re-running the same query refreshes AGENTS.md in-place.
"""

from __future__ import annotations

import logging
import os

from skills.registry import skill_library

logger = logging.getLogger(__name__)

_AGENTS_MD_GEN_SYSTEM_PROMPT = (
    "You are a technical architect assistant for an autonomous multi-agent coding harness.\n\n"
    "Your task is to write a comprehensive AGENTS.md file in Markdown format that is "
    "SPECIFICALLY tailored to a single user request. The coding harness will give this "
    "file to all three of its subagents (planning-agent, design-agent, build-agent) "
    "as their primary directive before they start working.\n\n"
    "Structure your AGENTS.md with these EXACT sections:\n\n"
    "## 1. Mission\n"
    "Describe precisely what is being built for this request (1 paragraph).\n\n"
    "## 2. Subagent Responsibilities\n"
    "- **planning-agent**: What scope/requirements/files it must plan for THIS feature.\n"
    "- **design-agent**: What architecture/interfaces/components to design.\n"
    "- **build-agent**: What to implement, test, and verify.\n\n"
    "## 3. Technical Constraints\n"
    "List languages, frameworks, security/perf requirements specific to this task.\n\n"
    "## 4. Acceptance Criteria\n"
    "A numbered list of concrete, testable conditions that define done for this task.\n\n"
    "## 5. Prohibited Actions\n"
    "Actions the agents must NOT take for this specific request.\n\n"
    "Rules: Be concise and actionable. Keep total output under 600 words. "
    "Do NOT include code blocks. Return raw Markdown only."
)


async def generate_agents_md(
    user_request: str,
    root_dir: str = ".",
    model: str | None = None,
) -> str:
    """LLM-generate a query-specific AGENTS.md and write it to harness/AGENTS.md.

    Args:
        user_request: The raw user query (e.g. "Add OAuth login to my app").
        root_dir:     Project workspace root, anchors the harness/ directory.
        model:        Optional LLM override. Resolved from AGENTS_MD_MODEL env var
                      or falls back to gpt-oss:120b-cloud via ChatOllama.

    Returns:
        The generated AGENTS.md content string, or empty string on failure.
    """
    agents_md_path = os.path.join(root_dir, "harness", "AGENTS.md")
    try:
        content = await _call_llm_for_agents_md(user_request, model=model)
        if not content or not content.strip():
            logger.warning("AgentsMDWriter: LLM returned empty content, skipping write.")
            return ""

        # Append any skills matched by the user query
        skills_context = skill_library.build_skills_context(user_request)
        if skills_context:
            content = content + "\n" + skills_context
            logger.info(
                f"AgentsMDWriter: injected {len(skill_library.select(user_request))} matched skill(s)."
            )

        _write_agents_md(agents_md_path, user_request, content)
        logger.info(f"AgentsMDWriter: harness/AGENTS.md written ({len(content)} chars).")
        return content
    except Exception as exc:
        logger.warning(
            f"AgentsMDWriter: Failed to generate AGENTS.md — {exc}. Keeping existing file."
        )
        return ""


async def _call_llm_for_agents_md(user_request: str, model: str | None = None) -> str:
    """Invokes the LLM to produce AGENTS.md content for the given user request."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm(model)
    messages = [
        SystemMessage(content=_AGENTS_MD_GEN_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User request:\n\n{user_request.strip()}\n\n"
                "Generate a concise, actionable AGENTS.md tailored to this exact request."
            )
        ),
    ]
    response = await llm.ainvoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def _build_llm(model: str | None = None) -> object:
    """Constructs a lightweight LLM instance for AGENTS.md generation.

    Resolution order:
      1. Explicit model argument.
      2. AGENTS_MD_MODEL environment variable.
      3. Falls back to ChatOllama with gpt-oss:120b-cloud.
    """
    resolved = model or os.getenv("AGENTS_MD_MODEL", "")

    if resolved.startswith("anthropic:") or "claude" in resolved:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=resolved.replace("anthropic:", ""),
            temperature=0.2,
            max_tokens=1024,
        )

    if resolved.startswith("openai:") or resolved.startswith("gpt-4") or resolved.startswith("gpt-3"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=resolved.replace("openai:", ""),
            temperature=0.2,
            max_tokens=1024,
        )

    # Default: local Ollama (matches the rest of the harness)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=resolved or "gpt-oss:120b-cloud", temperature=0.2)


def _write_agents_md(path: str, user_request: str, content: str) -> None:
    """Writes the generated content to harness/AGENTS.md with a metadata header."""
    import datetime

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    preview = user_request.strip()[:200].replace("\n", " ")
    header = (
        f"<!-- AUTO-GENERATED by AgentsMDWriter -->\n"
        f"<!-- Generated at: {ts} -->\n"
        f"<!-- User Request: {preview} -->\n\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + content + "\n")
