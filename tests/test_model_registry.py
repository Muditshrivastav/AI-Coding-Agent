import asyncio

import pytest

from agent.core import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    CodingAgentHarness,
    normalize_model_identifier,
)
from frameworks.model_config import create_chat_model
from nodes.build_subagent import build_dev_subagent
from tools.api_tools import APIToolset


def test_beam_cloud_is_the_default_and_registered_ollama_model():
    assert DEFAULT_MODEL == "ollama:beam:cloud"
    assert "ollama:beam:cloud" in SUPPORTED_MODELS
    assert all(model["provider"] == "ollama" for model in SUPPORTED_MODELS.values())
    assert SUPPORTED_MODELS["ollama:beam:cloud"]["provider"] == "ollama"
    assert SUPPORTED_MODELS["ollama:beam:cloud"]["name"] == "beam:cloud"


def test_beam_cloud_alias_normalizes_to_registered_model():
    assert normalize_model_identifier("beam:cloud") == "ollama:beam:cloud"


def test_groq_model_identifiers_are_rejected():
    with pytest.raises(ValueError, match="Only Ollama models are supported"):
        normalize_model_identifier("groq:old-model")


def test_groq_chat_model_construction_is_rejected():
    with pytest.raises(ValueError, match="Only Ollama models are supported"):
        create_chat_model("groq:old-model")


def test_groq_is_not_available_as_an_agent_tool():
    assert "groq_infer" not in {tool.name for tool in APIToolset().get_tools()}


def test_build_subagent_uses_harness_model_when_provided():
    spec = build_dev_subagent(all_tools=[], model="ollama:beam:cloud")

    assert spec["model"] == "ollama:beam:cloud"


def test_requested_model_is_used_by_root_and_each_subagent(monkeypatch):
    harness = CodingAgentHarness.__new__(CodingAgentHarness)
    harness._model = "ollama:stale-model"
    harness._agents_by_model = {}
    harness._planners_by_model = {}
    harness._subagents = [
        {"name": "planning-agent", "runnable": "old-planner"},
        {"name": "design-agent", "runnable": "old-designer"},
        {"name": "build-agent", "model": "ollama:stale-model", "tools": []},
    ]
    harness._root_dir = "."
    harness._guard = object()
    harness._drawio_client = object()
    captured = {}
    planner = object()
    designer = object()

    monkeypatch.setattr(
        "agent.core.make_planning_subagent",
        lambda **kwargs: captured.update(planner_model=kwargs["model"]) or planner,
    )

    async def make_design_subagent(**kwargs):
        captured["designer_model"] = kwargs["model"]
        return designer, harness._drawio_client

    monkeypatch.setattr("agent.core.make_design_subagent", make_design_subagent)

    def build_root(model, subagents=None):
        captured["root_model"] = model
        captured["subagents"] = subagents
        return "selected-agent"

    harness._build_root_agent = build_root

    agent = asyncio.run(harness.get_agent_for_model("ollama:beam:cloud"))

    assert agent == "selected-agent"
    assert captured["root_model"] == "ollama:beam:cloud"
    assert captured["planner_model"] == "ollama:beam:cloud"
    assert captured["designer_model"] == "ollama:beam:cloud"
    assert captured["subagents"][0]["runnable"] is planner
    assert captured["subagents"][1]["runnable"] is designer
    assert captured["subagents"][2]["model"] == "ollama:beam:cloud"
