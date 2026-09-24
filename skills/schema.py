"""
skills/schema.py - Skill data model and SkillLibrary registry.

A Skill is a named, tagged, reusable instruction block that the coding agent
injects into its context when a user request is relevant. This lets users
teach the agent domain-specific workflows, conventions, and best practices.

Lifecycle
---------
1. Skills are loaded from YAML files in skills/library/ at startup.
2. On each user query, SkillSelector finds relevant skills by keyword / tag match.
3. Matched skill instructions are appended to the auto-generated AGENTS.md.
4. The planning, design, and build subagents read these injected skills as guidelines.

Skill YAML Format (skills/library/<name>.yaml)
-----------------------------------------------
name: oauth
description: How to implement OAuth 2.0 login flows
tags: [auth, oauth, login, security]
instructions: |
  ## OAuth Implementation Guidelines
  - Use Authorization Code Flow with PKCE for web apps.
  - Store tokens in HttpOnly cookies, never localStorage.
  - Validate state parameter on callback to prevent CSRF.
examples:
  - "add google oauth to my login page"
  - "implement github oauth"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A single reusable skill that encodes domain-specific knowledge and conventions.

    Attributes:
        name:         Unique identifier slug (e.g. "oauth", "docker", "unit-testing").
        description:  Short human-readable summary shown in skill listings.
        tags:         Keywords for relevance matching against user queries.
        instructions: Full Markdown instruction block injected into AGENTS.md.
        examples:     Example user queries that should trigger this skill.
        priority:     Lower number = injected first when multiple skills match. Default 50.
        enabled:      Set False to disable without deleting. Default True.
        source:       Where this skill was loaded from ("builtin", "user", or a file path).
    """

    name: str
    description: str
    tags: list[str]
    instructions: str
    examples: list[str] = field(default_factory=list)
    priority: int = 50
    enabled: bool = True
    source: str = "user"

    def matches_query(self, query: str) -> bool:
        """Returns True if the query contains any tag or example keyword (case-insensitive)."""
        q = query.lower()
        return (
            any(tag.lower() in q for tag in self.tags)
            or any(kw.lower() in q for ex in self.examples for kw in ex.split())
        )

    def to_markdown_block(self) -> str:
        """Returns the skill formatted as a Markdown section for AGENTS.md injection."""
        return (
            f"---\n"
            f"### Skill: {self.name}\n"
            f"_{self.description}_\n\n"
            f"{self.instructions.strip()}\n"
        )


class SkillLibrary:
    """Registry of all available Skill objects.

    Provides:
    - register(skill): add a skill programmatically.
    - load_from_directory(path): bulk-load from YAML files.
    - get(name): retrieve a specific skill by name.
    - select(query): return all enabled skills relevant to a query, sorted by priority.
    - list_all(): list metadata for every registered skill.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a Skill instance. Overwrites any existing skill with the same name."""
        self._skills[skill.name] = skill
        logger.debug(f"SkillLibrary: registered skill '{skill.name}' (source={skill.source})")

    def get(self, name: str) -> Skill | None:
        """Retrieve a skill by name. Returns None if not found."""
        return self._skills.get(name)

    def remove(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it existed."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def select(self, query: str) -> list[Skill]:
        """Return all enabled skills that match the query, sorted by priority ascending."""
        matched = [
            s for s in self._skills.values()
            if s.enabled and s.matches_query(query)
        ]
        return sorted(matched, key=lambda s: s.priority)

    def list_all(self) -> list[dict[str, Any]]:
        """Return a lightweight summary list of all registered skills."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "tags": s.tags,
                "enabled": s.enabled,
                "priority": s.priority,
                "source": s.source,
            }
            for s in sorted(self._skills.values(), key=lambda s: s.priority)
        ]

    def load_from_directory(self, directory: str | Path) -> int:
        """Load all .yaml / .yml skill files from a directory.

        Returns:
            Number of skills successfully loaded.
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.warning("SkillLibrary: PyYAML not installed, cannot load from directory.")
            return 0

        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"SkillLibrary: skill directory not found: {dir_path}")
            return 0

        count = 0
        for yaml_file in sorted(dir_path.glob("*.y*ml")):
            try:
                with open(yaml_file, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict) or "name" not in data:
                    logger.warning(f"SkillLibrary: skipping invalid skill file: {yaml_file}")
                    continue
                skill = Skill(
                    name=data["name"],
                    description=data.get("description", ""),
                    tags=data.get("tags", []),
                    instructions=data.get("instructions", ""),
                    examples=data.get("examples", []),
                    priority=int(data.get("priority", 50)),
                    enabled=bool(data.get("enabled", True)),
                    source=str(yaml_file),
                )
                self.register(skill)
                count += 1
            except Exception as exc:
                logger.warning(f"SkillLibrary: failed to load {yaml_file}: {exc}")

        logger.info(f"SkillLibrary: loaded {count} skill(s) from {dir_path}")
        return count

    def build_skills_context(self, query: str) -> str:
        """Build a Markdown block of all matched skills to inject into AGENTS.md.

        Returns empty string if no skills match.
        """
        matched = self.select(query)
        if not matched:
            return ""
        header = "\n---\n## Active Skills\n\n_The following skills have been automatically selected for this request:_\n\n"
        body = "\n\n".join(s.to_markdown_block() for s in matched)
        return header + body + "\n"
