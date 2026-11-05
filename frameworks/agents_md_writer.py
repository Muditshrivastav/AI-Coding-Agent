"""Generate request-scoped TASK.md without modifying permanent AGENTS.md rules."""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

from skills.registry import skill_library

logger = logging.getLogger(__name__)

_TASK_MD_GEN_SYSTEM_PROMPT = (
    "You prepare concise, actionable task instructions for an autonomous coding agent. "
    "Write TASK.md for only the current user request. Include its goal, scope, constraints, "
    "and testable acceptance criteria. Do not copy permanent repository rules, invent files, "
    "or include skills or prior task content. Keep the result under 600 words and return raw "
    "Markdown without code fences."
)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "app",
    "build",
    "create",
    "for",
    "from",
    "help",
    "implement",
    "in",
    "make",
    "me",
    "of",
    "on",
    "please",
    "the",
    "to",
    "update",
    "with",
}


def _relevant_skill_summaries(
    user_request: str, *, threshold: float = 0.2
) -> list[str]:
    request_terms = {
        term
        for term in re.findall(r"[a-z0-9+#.-]+", user_request.lower())
        if term not in _STOP_WORDS
    }
    relevant: list[tuple[float, str]] = []
    for skill in skill_library.select(user_request):
        keywords = {
            term
            for value in [skill.name, *skill.tags]
            for term in re.findall(r"[a-z0-9+#.-]+", value.lower())
        } - _STOP_WORDS
        if not keywords:
            continue
        score = len(request_terms & keywords) / len(keywords)
        if score >= threshold:
            summary = skill.description.strip() or skill.name
            relevant.append((score, f"- **{skill.name}**: {summary}"))
    relevant.sort(key=lambda item: (-item[0], item[1]))
    return [summary for _, summary in relevant[:3]]


async def generate_agents_md(
    user_request: str,
    root_dir: str = ".",
    model: str | None = None,
) -> str:
    """Generate request-specific instructions in ``harness/TASK.md``.

    The historical function name is retained for existing callers. The
    permanent ``harness/AGENTS.md`` is never read or overwritten.
    """
    task_path = os.path.join(root_dir, "harness", "TASK.md")
    try:
        content = await _call_llm_for_agents_md(user_request, model=model)
        if not content or not content.strip():
            logger.warning(
                "Task writer: LLM returned empty content; TASK.md was not changed."
            )
            return ""

        summaries = _relevant_skill_summaries(user_request)
        if summaries:
            content = (
                content.rstrip() + "\n\n## Relevant skills\n\n" + "\n".join(summaries)
            )
        _write_task_md(task_path, content)
        logger.info("Task writer: harness/TASK.md written (%s chars).", len(content))
        return content
    except Exception as exc:
        logger.warning("Task writer: failed to generate harness/TASK.md: %s", exc)
        return ""


async def _call_llm_for_agents_md(
    user_request: str,
    model: str | None = None,
) -> str:
    """Generate only the current request; previous TASK.md content is not context."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm(model)
    response = await llm.ainvoke(
        [
            SystemMessage(content=_TASK_MD_GEN_SYSTEM_PROMPT),
            HumanMessage(content=f"Current user request:\n\n{user_request.strip()}"),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part if isinstance(part, str) else part["text"]
        for part in content
        if isinstance(part, str)
        or (isinstance(part, dict) and isinstance(part.get("text"), str))
    )


def _build_llm(model: str | None = None) -> BaseChatModel:
    """Construct the task writer model using the shared model configuration."""
    from frameworks.model_config import task_writer_model

    return task_writer_model(model)


def _write_task_md(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content.rstrip() + "\n")
