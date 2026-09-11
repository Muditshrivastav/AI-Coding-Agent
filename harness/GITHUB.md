# GitHub Automation & Repository Workflow Instructions

This document instructs the **build-agent** on how to initialize, stage, commit, and publish the user's project into a remote GitHub repository using commands executed via `LocalShellBackend`.

---

## 1. Safety & Guardrails Protocol

Before running any GitHub command via `LocalShellBackend`, observe the following rules:

1. **Verify Sandbox Boundary**: All commands must execute within `root_dir`.
2. **Respect `harness/permissions.json`**:
   - `git add`, `git commit`, `git status`, `git branch` are allowed.
   - `git push` triggers the **HITL (Human-in-the-Loop) approval gate** (`ask` permission). Do not bypass or force push without approval.
3. **Sensitive Files Exclusion**:
   - Verify `.gitignore` exists and excludes `.env`, `secrets/`, `*.key`, `__pycache__/`, `.venv/`, and IDE config directories.
   - Never stage `.env` or files matching `deny` rules.

---

## 2. Step-by-Step GitHub Publish Workflow

Execute these commands sequentially via `LocalShellBackend`:

### Step 1: Pre-flight Checks & Verification
Ensure all code passes verification before any git staging:
```bash
# 1. Run local verification
make verify
# or invoke the run_verification tool
```

Check current git status and branch:
```bash
git status
```

---

### Step 2: Initialize Git Repository (if not already initialized)
If the project directory is not yet a git repository:
```bash
git init
git branch -M main
```

Ensure standard `.gitignore` is present:
```bash
# Verify .gitignore content includes sensitive patterns
cat .gitignore
```

---

### Step 3: Configure Git User (if not globally configured)
```bash
git config user.name "ai-coding-agent"
git config user.email "agent@autonomous.local"
```

---

### Step 4: Stage and Commit Changes
Stage files cleanly (excluding untracked secrets):
```bash
git add .
git status
```

Create a semantic commit following conventional commits format:
```bash
git commit -m "feat: implement project features based on PLAN.md and ARCHITECTURE.md"
```

---

### Step 5: Configure Remote Repository
Check existing remotes:
```bash
git remote -v
```

If no remote exists, add the user's GitHub repository:
```bash
# Using HTTPS with GITHUB_PAT from environment
git remote add origin https://${GITHUB_PAT}@github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git

# Or if remote origin already exists:
git remote set-url origin https://${GITHUB_PAT}@github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git
```

*Note: Alternatively, if `gh` (GitHub CLI) is available and authenticated:*
```bash
gh repo create ${GITHUB_REPO} --public --source=. --remote=origin --push
```

---

### Step 6: Create Feature Branch (Recommended) or Push to Main
Create a feature branch for review:
```bash
git checkout -b feature/initial-implementation
```

Push to remote (requires HITL approval as per `permissions.json`):
```bash
git push -u origin feature/initial-implementation
```

Or push directly to `main`:
```bash
git push -u origin main
```

---

### Step 7: Create Pull Request (Optional via GitHub MCP or `gh` CLI)
If targeting a branch PR:
```bash
# Via GitHub CLI:
gh pr create --title "feat: user project implementation" --body "Automated implementation based on PLAN.md and ARCHITECTURE.md. All tests verified via make verify." --base main --head feature/initial-implementation
```
*Or use the bound `mcp__github__create_pull_request` MCP tool.*

---

## 3. Error Recovery & Common Gotchas

- **Remote Already Exists**: Use `git remote set-url origin <URL>`.
- **Merge Conflicts**: Execute `git fetch origin main` followed by `git rebase origin/main`.
- **Large/Binary Files**: Ensure `.gitignore` ignores large checkpoints and weights.
- **Authentication Failure**: Check whether `GITHUB_PAT` or SSH keys are properly loaded in the session environment.
