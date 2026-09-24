"""
skills/registry.py - Global SkillLibrary singleton with auto-loaded built-in skills.

Usage
-----
    from skills.registry import skill_library

    # Register a custom skill at runtime
    from skills.schema import Skill
    skill_library.register(Skill(
        name="my-convention",
        description="Our team's coding conventions",
        tags=["python", "style"],
        instructions="Always use Black formatter. Max line length 88.",
    ))

    # Load user-provided skill YAML files from a directory
    skill_library.load_from_directory("path/to/my/skills/")

    # Get skills matching a query
    matched = skill_library.select("add oauth login to my app")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from skills.schema import Skill, SkillLibrary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton — import this everywhere instead of instantiating your own.
# ---------------------------------------------------------------------------
skill_library = SkillLibrary()

# ---------------------------------------------------------------------------
# Auto-load built-in skills from skills/library/
# ---------------------------------------------------------------------------
_BUILTIN_SKILLS_DIR = Path(__file__).parent / "library"

_loaded = skill_library.load_from_directory(_BUILTIN_SKILLS_DIR)
if _loaded == 0:
    logger.debug("SkillLibrary: no built-in YAML skills loaded (PyYAML may not be installed).")

# ---------------------------------------------------------------------------
# Auto-load user-provided skills from SKILL_LIBRARY_PATH env var
# ---------------------------------------------------------------------------
_user_skills_path = os.getenv("SKILL_LIBRARY_PATH", "")
if _user_skills_path and Path(_user_skills_path).is_dir():
    _user_loaded = skill_library.load_from_directory(_user_skills_path)
    logger.info(f"SkillLibrary: loaded {_user_loaded} user skill(s) from {_user_skills_path}")
