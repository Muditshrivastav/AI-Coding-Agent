import asyncio

from frameworks.agents_md_writer import generate_agents_md


def test_task_writer_keeps_permanent_agents_and_replaces_only_task_file(
    tmp_path, monkeypatch
):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    agents_file = harness_dir / "AGENTS.md"
    agents_file.write_text("Permanent repository rules", encoding="utf-8")
    task_file = harness_dir / "TASK.md"
    task_file.write_text("Previous request content", encoding="utf-8")
    captured_requests = []

    async def generate(request, model=None):
        captured_requests.append(request)
        return "## Current task\nImplement the requested feature."

    monkeypatch.setattr(
        "frameworks.agents_md_writer._call_llm_for_agents_md",
        generate,
    )

    result = asyncio.run(
        generate_agents_md("Implement a new feature", root_dir=str(tmp_path))
    )

    assert captured_requests == ["Implement a new feature"]
    assert "Implement a new feature" == captured_requests[0]
    assert "Previous request content" not in result
    assert task_file.read_text(encoding="utf-8") == result + "\n"
    assert agents_file.read_text(encoding="utf-8") == "Permanent repository rules"
