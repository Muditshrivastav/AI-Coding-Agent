"""
frameworks/langgraph_runtime.py - LangGraph runtime and checkpointer management.
"""

from typing import Any
from langgraph.checkpoint.memory import MemorySaver


class LangGraphRuntime:
    """Manages LangGraph checkpoints and execution state."""

    def __init__(self, checkpointer: Any = None) -> None:
        self._checkpointer = checkpointer or MemorySaver()

    @property
    def checkpointer(self) -> Any:
        return self._checkpointer

    @classmethod
    def create_memory_saver(cls) -> MemorySaver:
        return MemorySaver()
