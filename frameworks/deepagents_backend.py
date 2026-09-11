"""
frameworks/deepagents_backend.py - Deep Agents integration wrapper.
Manages LocalShellBackend scoped to root_dir and builds deep agents with subagents.
"""

from typing import Any
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend


class DeepAgentsBackend:
    """Wraps Deep Agents library, providing a root_dir scoped backend and agent constructor."""

    def __init__(self, root_dir: str = ".") -> None:
        self._root_dir = root_dir
        self._backend = LocalShellBackend(root_dir=root_dir)

    @property
    def backend(self) -> LocalShellBackend:
        return self._backend

    def build_agent(
        self,
        model: str,
        subagents: list[Any],
        system_prompt: str | None = None,
        checkpointer: Any = None,
        tools: list[Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "backend": self._backend,
            "subagents": subagents,
            "checkpointer": checkpointer,
        }
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        return create_deep_agent(**kwargs)
