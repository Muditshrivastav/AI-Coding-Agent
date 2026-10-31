"""Bounded LangGraph orchestration for repairing failed agent tool runs."""

from __future__ import annotations

import json
import re
from typing import Any, Required, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


_COMMAND_FAILURE = re.compile(r"\[Command failed with exit code [1-9]\d*\]", re.IGNORECASE)
_DIAGNOSTIC_FAILURE = re.compile(
    r"verification.{0,30}failed|tests? (?:have )?failed|test failure|"
    r"lint.{0,30}failed|build.{0,30}failed|\b\d+ failed\b|"
    r"traceback \(most recent call last\)",
    re.IGNORECASE,
)


class RepairLoopState(TypedDict, total=False):
    user_request: Required[str]
    conversation_history: str
    execution_thread_id: Required[str]
    repair_round: int
    max_repair_rounds: int
    repair_feedback: str
    last_output: str
    final_output: str
    failures: list[str]
    status: str


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        return " ".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _tool_failure(status: str, content: str) -> bool:
    if status == "error":
        return True
    return bool(_COMMAND_FAILURE.search(content) or _DIAGNOSTIC_FAILURE.search(content))


def build_repair_graph(
    agent: Any,
    *,
    checkpointer: Any = None,
    max_repair_rounds: int = 3,
) -> Any:
    """Run the agent again with failure diagnostics, stopping at a fixed retry cap."""
    if max_repair_rounds < 0:
        raise ValueError("max_repair_rounds must be non-negative")

    async def execute_round(
        state: RepairLoopState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        writer = get_stream_writer()
        round_number = state.get("repair_round", 0)
        writer({"type": "iteration_start", "iteration": round_number + 1})

        prompt = state.get("repair_feedback") or state["user_request"]
        history = state.get("conversation_history", "").strip()
        if history and round_number == 0:
            prompt = f"Conversation history:\n{history}\n\nCurrent request:\n{prompt}"

        child_thread_id = state["execution_thread_id"]
        child_config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": child_thread_id,
            },
        }
        latest_tool_results: dict[str, tuple[str, str]] = {}
        output = ""

        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=child_config,
            stream_mode="updates",
        ):
            if not isinstance(chunk, dict):
                continue
            for update in chunk.values():
                messages = update.get("messages", []) if isinstance(update, dict) else []
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    message_type = getattr(message, "type", "")
                    if message_type == "tool":
                        name = getattr(message, "name", "tool") or "tool"
                        content = _message_text(getattr(message, "content", ""))
                        latest_tool_results[name] = (content, getattr(message, "status", ""))
                        writer({
                            "type": "tool_result",
                            "tool": name,
                            "output": content.splitlines()[0][:120] if content else "",
                            "id": getattr(message, "tool_call_id", "") or "",
                            "lines": content.count("\n") + 1 if content else 0,
                        })
                    elif message_type == "ai":
                        tool_calls = getattr(message, "tool_calls", []) or []
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                name = tool_call.get("name", "tool")
                                arguments = tool_call.get("args", {})
                                call_id = tool_call.get("id", "")
                            else:
                                name = getattr(tool_call, "name", "tool")
                                arguments = getattr(tool_call, "args", {})
                                call_id = getattr(tool_call, "id", "")
                            writer({
                                "type": "tool_call",
                                "tool": name,
                                "input": json.dumps(arguments, default=str)[:120],
                                "id": call_id,
                            })
                        content = _message_text(getattr(message, "content", ""))
                        if content.strip() and not tool_calls:
                            output = content.strip()
                            writer({"type": "text_delta", "text": content})

        failures = [
            f"{name}: {content[:2000]}"
            for name, (content, status) in latest_tool_results.items()
            if _tool_failure(status, content)
        ]
        writer({
            "type": "iteration_complete",
            "iteration": round_number + 1,
            "failed": bool(failures),
        })

        result: dict[str, Any] = {
            "last_output": output,
            "failures": failures,
            "final_output": output or "Task completed.",
        }
        if failures and round_number >= state.get("max_repair_rounds", max_repair_rounds):
            details = "\n".join(f"- {failure}" for failure in failures)
            result.update({
                "status": "needs_attention",
                "final_output": (
                    f"{output}\n\n"
                    if output
                    else ""
                ) + (
                    "⚠️ Automatic repair limit reached before the failing checks passed.\n\n"
                    f"Latest failure details:\n{details}"
                ),
            })
        elif not failures:
            result["status"] = "complete"
        return result

    def prepare_repair(state: RepairLoopState) -> dict[str, Any]:
        next_round = state.get("repair_round", 0) + 1
        details = "\n".join(f"- {failure}" for failure in state.get("failures", []))
        feedback = (
            "The previous work round reported failures. Continue working in the same "
            "workspace, diagnose and fix these exact issues, then rerun the relevant "
            "verification. Do not merely explain the failures.\n\n"
            f"Failure details:\n{details}\n\n"
            f"Previous response:\n{state.get('last_output', '')}"
        )
        get_stream_writer()({
            "type": "repair_started",
            "iteration": next_round + 1,
            "failures": state.get("failures", []),
        })
        return {"repair_round": next_round, "repair_feedback": feedback}

    def route_after_round(state: RepairLoopState) -> str:
        if not state.get("failures"):
            return END
        if state.get("repair_round", 0) >= state.get("max_repair_rounds", max_repair_rounds):
            return END
        return "prepare_repair"

    graph = StateGraph(RepairLoopState)
    graph.add_node("execute_round", execute_round)
    graph.add_node("prepare_repair", prepare_repair)
    graph.add_edge(START, "execute_round")
    graph.add_conditional_edges("execute_round", route_after_round)
    graph.add_edge("prepare_repair", "execute_round")
    return graph.compile(checkpointer=checkpointer)
