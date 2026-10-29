import asyncio

from deepagents import create_deep_agent as build_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from nodes.design_subagent import make_design_subagent


def test_design_subagent_can_read_and_edit_workspace_files(tmp_path, monkeypatch):
    class FakeDrawioClient:
        async def get_tools(self):
            return []

    model = FakeListChatModel(responses=["done"])
    captured = {}

    def create_agent(**kwargs):
        assert isinstance(kwargs["backend"], FilesystemBackend)
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["backend"] = kwargs["backend"]
        return build_deep_agent(**{**kwargs, "model": model})

    monkeypatch.setattr("nodes.design_subagent.ChatGroq", lambda **kwargs: model)
    monkeypatch.setattr("nodes.design_subagent.ChatOllama", lambda **kwargs: model)
    monkeypatch.setattr("nodes.design_subagent.create_deep_agent", create_agent)
    monkeypatch.setattr(
        "nodes.design_subagent.drawio_mcp_client", FakeDrawioClient
    )

    file_path = tmp_path / "notes.py"
    file_path.write_text("before\n", encoding="utf-8")

    agent, _ = asyncio.run(make_design_subagent(root_dir=str(tmp_path)))
    tools = agent.nodes["tools"].bound.tools_by_name

    assert "read_file" in tools
    assert "edit_file" in tools
    assert "workspace root is mounted at `/`" in captured["system_prompt"]
    assert captured["backend"].read("/notes.py").file_data["content"] == "before\n"

    captured["backend"].edit("/notes.py", "before", "after")
    assert file_path.read_text(encoding="utf-8") == "after\n"
