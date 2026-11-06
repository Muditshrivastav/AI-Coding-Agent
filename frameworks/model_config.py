"""Central model identifiers and constructors for agent components."""

from __future__ import annotations

import os
from typing import Any

OLLAMA_DEFAULT_MODEL = "ollama:beam:cloud"
OLLAMA_FALLBACK_MODEL = os.getenv(
    "AGENT_FALLBACK_MODEL",
    "ollama:gpt-oss:120b-cloud",
)
DEFAULT_MODEL = os.getenv("AGENT_MODEL", OLLAMA_DEFAULT_MODEL)
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
        raise ValueError(f"Model '{model_id}' must include an Ollama provider prefix.")
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=name, temperature=temperature, **kwargs)
    raise ValueError(
        f"Unsupported model provider '{provider}' in '{model_id}'. "
        "Only Ollama models are supported."
    )


def task_writer_model(model_id: str | None = None) -> Any:
    return create_chat_model(model_id or TASK_WRITER_MODEL, temperature=0.2)
