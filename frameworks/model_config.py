"""Central model identifiers and constructors for agent components."""

from __future__ import annotations

import os
from typing import Any

GROQ_DEFAULT_MODEL = os.getenv(
    "GROQ_AGENT_MODEL",
    "groq:qwen/qwen3.8-27b",
)
OLLAMA_FALLBACK_MODEL = os.getenv(
    "AGENT_FALLBACK_MODEL",
    "ollama:gpt-oss:120b-cloud",
)
DEFAULT_MODEL = os.getenv("AGENT_MODEL", GROQ_DEFAULT_MODEL)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", DEFAULT_MODEL)
DESIGNER_MODEL = os.getenv("DESIGNER_MODEL", PLANNER_MODEL)
BUILDER_MODEL = os.getenv("BUILDER_MODEL", DEFAULT_MODEL)
TASK_WRITER_MODEL = os.getenv("TASK_WRITER_MODEL", PLANNER_MODEL)
FALLBACK_MODEL = OLLAMA_FALLBACK_MODEL


def create_chat_model(
    model_id: str,
    *,
    temperature: float = 0.2,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault("max_retries", 2)
    provider, separator, name = model_id.partition(":")
    if not separator:
        provider, name = "groq", model_id
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=name, temperature=temperature, **kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, temperature=temperature, **kwargs)
    raise ValueError(f"Unsupported model provider in '{model_id}'.")


def task_writer_model(model_id: str | None = None) -> Any:
    return create_chat_model(model_id or TASK_WRITER_MODEL, temperature=0.2)
