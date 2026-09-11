"""
tests/test_harness_guard.py - Unit tests for permissions.json guardrails.
"""

from agent.harness import HarnessGuard


def test_harness_guard_allow_rules():
    guard = HarnessGuard("harness/permissions.json")
    assert guard.check_permission("Bash(ruff check .)") == "allow"
    assert guard.check_permission("Bash(pytest -q)") == "allow"
    assert guard.check_permission("Read(src/app.py)") == "allow"


def test_harness_guard_deny_rules():
    guard = HarnessGuard("harness/permissions.json")
    assert guard.check_permission("Read(.env)") == "deny"
    assert guard.check_permission("Read(secrets/key.pem)") == "deny"
    assert guard.check_permission("Bash(rm -rf /)") == "deny"


def test_harness_guard_ask_rules():
    guard = HarnessGuard("harness/permissions.json")
    assert guard.check_permission("Bash(git push origin main)") == "ask"
    assert guard.check_permission("vercel_deploy") == "ask"
    assert guard.check_permission("mcp__github__create_pull_request") == "ask"
