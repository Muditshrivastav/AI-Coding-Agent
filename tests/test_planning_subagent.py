import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from nodes.planning_subagent import make_planning_subagent


def test_planning_subagent_registers_plan_and_task_tools(tmp_path, monkeypatch):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "permissions.json").write_text(
        json.dumps({"allow": ["Read(*)"], "ask": [], "deny": []}),
        encoding="utf-8",
    )
    model = FakeListChatModel(responses=["Plan created."])
    monkeypatch.setattr(
        "nodes.planning_subagent.create_chat_model",
        lambda *args, **kwargs: model,
    )

    planner = make_planning_subagent(root_dir=str(tmp_path))
    tools = planner.nodes["tools"].bound.tools_by_name

    assert {"write_plan", "write_task_plan"} <= set(tools)
