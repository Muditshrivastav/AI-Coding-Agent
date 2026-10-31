import asyncio
from types import SimpleNamespace

import pytest

from agent.repair_loop import build_repair_graph


class FakeAgent:
    def __init__(self, tool_results):
        self.tool_results = tool_results
        self.prompts = []

    async def astream(self, payload, *, config, stream_mode):
        self.prompts.append(payload["messages"][-1]["content"])
        result = self.tool_results[min(len(self.prompts) - 1, len(self.tool_results) - 1)]
        messages = [
            SimpleNamespace(
                type="tool",
                name=result["name"],
                content=result["content"],
                status=result.get("status", ""),
                tool_call_id="call-1",
            ),
            SimpleNamespace(type="ai", content=result.get("answer", ""), tool_calls=[]),
        ]
        yield {"agent": {"messages": messages}}


def _initial_state():
    return {
        "user_request": "Fix the failing tests",
        "execution_thread_id": "test-execution",
        "repair_round": 0,
        "max_repair_rounds": 3,
    }


def test_repair_loop_retries_failed_verification_and_stops_when_passed():
    agent = FakeAgent([
        {
            "name": "build_subagent",
            "content": "Build agent report: Verification FAILED: assertion mismatch",
            "answer": "I will fix it.",
        },
        {
            "name": "run_verification",
            "content": "✅ Verification PASSED",
            "answer": "Fixed and verified.",
        },
    ])
    graph = build_repair_graph(agent)

    result = asyncio.run(graph.ainvoke(
        _initial_state(),
        config={"configurable": {"thread_id": "repair-test"}},
    ))

    assert len(agent.prompts) == 2
    assert "assertion mismatch" in agent.prompts[1]
    assert result["status"] == "complete"
    assert result["final_output"] == "Fixed and verified."


def test_repair_loop_stops_after_successful_first_round():
    agent = FakeAgent([
        {
            "name": "run_verification",
            "content": "✅ Verification PASSED",
            "answer": "No changes were needed.",
        },
    ])
    graph = build_repair_graph(agent)

    result = asyncio.run(graph.ainvoke(
        _initial_state(),
        config={"configurable": {"thread_id": "repair-success"}},
    ))

    assert len(agent.prompts) == 1
    assert result["status"] == "complete"


def test_repair_loop_streams_tool_events_and_final_state():
    agent = FakeAgent([
        {
            "name": "run_verification",
            "content": "✅ Verification PASSED",
            "answer": "Verified successfully.",
        },
    ])
    graph = build_repair_graph(agent)

    async def collect():
        events = []
        final_state = None
        async for mode, chunk in graph.astream(
            _initial_state(),
            config={"configurable": {"thread_id": "repair-stream"}},
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                events.append(chunk)
            elif mode == "values":
                final_state = chunk
        return events, final_state

    events, final_state = asyncio.run(collect())

    assert any(event.get("type") == "tool_result" for event in events)
    assert any(event.get("type") == "text_delta" for event in events)
    assert final_state is not None
    assert final_state.get("status") == "complete"


def test_repair_loop_resumes_after_human_approval_interrupt():
    from typing import TypedDict

    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    class AgentState(TypedDict, total=False):
        messages: list

    def request_approval(state):
        interrupt({"description": "Approve the protected action"})
        return {
            "messages": [
                AIMessage(content="Action approved.")
            ]
        }

    child_builder = StateGraph(AgentState)
    child_builder.add_node("approval", request_approval)
    child_builder.add_edge(START, "approval")
    child_builder.add_edge("approval", END)
    checkpointer = MemorySaver()
    child_agent = child_builder.compile(checkpointer=checkpointer)

    class InterruptingAgent:
        async def astream(self, payload, *, config, stream_mode):
            async for chunk in child_agent.astream(
                payload,
                config=config,
                stream_mode=stream_mode,
            ):
                yield chunk

    graph = build_repair_graph(InterruptingAgent(), checkpointer=checkpointer)

    async def execute_and_resume():
        config = {"configurable": {"thread_id": "repair-interrupt"}}
        paused_state = {}
        interrupted = False
        try:
            async for mode, chunk in graph.astream(
                _initial_state(),
                config=config,
                stream_mode=["custom", "values"],
            ):
                if mode == "values":
                    paused_state = chunk
        except Exception as exc:
            if type(exc).__name__ not in ("GraphInterrupt", "NodeInterrupt"):
                raise
            interrupted = True
        assert interrupted or paused_state.get("__interrupt__")
        return await graph.ainvoke(Command(resume={"approved": True}), config=config)

    result = asyncio.run(execute_and_resume())

    assert result["status"] == "complete"
    assert result["final_output"] == "Action approved."


def test_repair_loop_terminates_at_retry_limit_with_failure_details():
    failure = {
        "name": "run_verification",
        "content": "❌ Verification FAILED: unresolved regression",
        "answer": "Still investigating.",
    }
    agent = FakeAgent([failure])
    graph = build_repair_graph(agent, max_repair_rounds=1)

    result = asyncio.run(graph.ainvoke(
        {**_initial_state(), "max_repair_rounds": 1},
        config={"configurable": {"thread_id": "repair-limit"}},
    ))

    assert len(agent.prompts) == 2
    assert result["status"] == "needs_attention"
    assert "retry limit" in result["final_output"].lower() or "repair limit" in result["final_output"].lower()
    assert "unresolved regression" in result["final_output"]


def test_repair_loop_rejects_negative_retry_limit():
    with pytest.raises(ValueError, match="non-negative"):
        build_repair_graph(FakeAgent([]), max_repair_rounds=-1)
