# Sandbox Execution Lifecycle — Implementation Spec

**Purpose**: Defines how a hosted, multi-tenant coding agent runs shell commands and edits code in isolation, while still being able to make real changes to a user's actual GitHub repository — without the sandbox ever having direct access to that real repository.

**Core principle**: the sandbox never touches the user's real code. It only ever touches a disposable, ephemeral local copy. Exactly one explicit, separate step (`git push`) transfers verified changes from that copy to the real repository. Everything before that step is fully reversible by simply discarding the copy; nothing before it is visible to the user.

---

## 1. Why a hosted agent needs this, specifically

For a coding agent that runs locally on a developer's own machine (e.g. a CLI tool), "the real directory" already exists on disk, and the sandbox can bind-mount it directly. For a **hosted, multi-tenant backend** — where the agent runs on shared server infrastructure serving many users — there is no local directory to bind-mount. The user's code lives entirely in their GitHub repository. This means the lifecycle needs an explicit step to bring a copy of that repository onto the server before any work can happen, and an explicit step to push changes back.

---

## 2. The five-stage lifecycle

```
1. CLONE   — pull the user's real repo into an isolated, ephemeral
             directory on the server, authenticated via a scoped,
             short-lived credential (e.g. a GitHub App installation token)

2. EXECUTE — bind-mount that clone (not the real repo, not the host
             machine) into a sandboxed container; all shell commands,
             file writes, and builds happen against this clone only,
             inside network/filesystem/resource restrictions

3. VERIFY  — run the project's checks (lint, tests, build) inside the
             sandbox, against the clone

4. PUSH    — ONLY IF verification passed: commit and push the clone's
             changes back to the user's real repository, using a
             freshly minted credential (the earlier one may have expired)

5. CLEANUP — delete the ephemeral clone from the server; nothing
             persists between runs
```

**The critical property**: stages 1–3 can run entirely inside a security boundary with zero effect on anything the user owns. Stage 4 is the only stage with an external effect, and it is gated on stage 3 succeeding. Stage 5 ensures no artifact of the run lingers on the server after it ends.

---

## 3. What is, and is not, exposed to the sandbox

| | Exposed to the sandboxed container | NOT exposed |
|---|---|---|
| The disposable clone (`run_dir`) | Yes — bind-mounted, read/write | — |
| The user's real GitHub repository | — | Never — the sandbox has no network path to GitHub at all |
| The host server's filesystem outside `run_dir` | — | Never — container filesystem isolation |
| Credentials (installation tokens) | — | Never — tokens are used by the orchestrator process outside the container, never passed into the sandboxed shell |
| Arbitrary internet access | — | Restricted to an explicit allowlist (package registries, not arbitrary hosts) |

This is why "the sandbox is isolated" and "the agent can still push real code" are not in tension: the isolation applies to the *copy*, and the *push* is a deliberate action taken by code running outside the sandbox entirely, after the sandbox's work is done and verified.

---

## 4. Implementation

### 4.1 Lifecycle manager — clone, push, cleanup

```python
# agent/run_lifecycle.py
import asyncio, shutil, uuid, os

class ProjectRunLifecycle:
    def __init__(self, github_app_auth, base_dir: str = "/var/agent-runs"):
        self._github_auth = github_app_auth
        self._base_dir = base_dir

    async def _run_git(self, *args, cwd: str = None):
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {stderr.decode()}")
        return stdout.decode()

    async def start_run(self, installation_id: str, repo_full_name: str) -> dict:
        """Stage 1: CLONE. Creates an isolated, disposable copy of the real repo."""
        run_id = str(uuid.uuid4())
        run_dir = os.path.join(self._base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        token = await self._github_auth.get_installation_token(installation_id)
        clone_url = f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
        await self._run_git("clone", "--depth", "1", clone_url, run_dir)

        return {"run_id": run_id, "run_dir": run_dir}

    async def finish_run(self, run_dir: str, installation_id: str, repo_full_name: str, branch: str = "main"):
        """Stage 4: PUSH. The ONLY step that affects the user's real repository.
        Call only after verification has passed."""
        token = await self._github_auth.get_installation_token(installation_id)  # fresh — earlier one may have expired
        push_url = f"https://x-access-token:{token}@github.com/{repo_full_name}.git"
        await self._run_git("remote", "set-url", "origin", push_url, cwd=run_dir)
        await self._run_git("push", "origin", branch, cwd=run_dir)

    def cleanup(self, run_dir: str):
        """Stage 5: CLEANUP. Deletes the disposable copy. Never deletes anything
        in the user's real repository — this only removes the local clone."""
        shutil.rmtree(run_dir, ignore_errors=True)
```

### 4.2 Harness integration — clone → sandbox → verify-gated push → cleanup

```python
# agent/core.py
class CodingAgentHarness:
    def __init__(self, github_app_auth, model: str, checkpointer=None):
        self._github_auth = github_app_auth
        self._model = model
        self._checkpointer = checkpointer
        self._lifecycle = ProjectRunLifecycle(github_app_auth)

    async def run(self, user_request: str, thread_id: str,
                   installation_id: str, repo_full_name: str, branch: str = "main"):
        run_info = await self._lifecycle.start_run(installation_id, repo_full_name)  # Stage 1
        run_dir = run_info["run_dir"]
        result = None

        try:
            backend = SandboxBackend(project_dir=run_dir)  # Stage 2 — bind-mounts the CLONE only
            agent = build_agent(model=self._model, backend=backend, checkpointer=self._checkpointer)

            config = {"configurable": {"thread_id": thread_id}}
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_request}]}, config=config
            )  # includes Stage 3 verification internally, as part of the build loop

            if result.get("__interrupt__"):
                # A human-in-the-loop pause: keep run_dir alive, do NOT clean up yet,
                # since resuming needs the same clone.
                return {"status": "awaiting_approval", "interrupt": result["__interrupt__"],
                        "run_dir": run_dir, "thread_id": thread_id}

            if result.get("verify_passed", True):
                await self._lifecycle.finish_run(run_dir, installation_id, repo_full_name, branch)  # Stage 4
                return {"status": "complete", "result": result}
            else:
                # Verification failed — Stage 4 is skipped entirely.
                # The real repository is untouched, exactly as if this run never happened.
                return {"status": "failed_verification", "result": result}

        except Exception as e:
            return {"status": "error", "error": str(e)}

        finally:
            should_cleanup = True
            if result is not None and result.get("__interrupt__"):
                should_cleanup = False  # keep the clone alive across a pending approval
            if should_cleanup:
                self._lifecycle.cleanup(run_dir)  # Stage 5

    async def resume(self, agent, thread_id: str, run_dir: str, approved: bool,
                      installation_id: str, repo_full_name: str, branch: str = "main"):
        config = {"configurable": {"thread_id": thread_id}}
        result = await agent.ainvoke(Command(resume={"approved": approved}), config=config)

        if not result.get("__interrupt__"):
            if result.get("verify_passed", True):
                await self._lifecycle.finish_run(run_dir, installation_id, repo_full_name, branch)
            self._lifecycle.cleanup(run_dir)
            return {"status": "complete", "result": result}

        return {"status": "awaiting_approval", "interrupt": result["__interrupt__"],
                "run_dir": run_dir, "thread_id": thread_id}
```

### 4.3 Persisting run state across a HITL pause

Because a paused run keeps its clone on disk, `run_dir` must be recoverable even if the server restarts between "approval requested" and "approval given." Store it alongside the thread, not only in memory:

```python
# on interrupt:
await db.save("runs", thread_id, {
    "run_dir": run_dir,
    "installation_id": installation_id,
    "repo_full_name": repo_full_name,
    "branch": branch,
})

# on resume:
run_state = await db.load("runs", thread_id)
result = await harness.resume(agent, thread_id, run_state["run_dir"], approved,
                               run_state["installation_id"], run_state["repo_full_name"], run_state["branch"])
```

---

## 5. Isolation boundaries inside Stage 2 (what the sandbox itself restricts)

Independent of the clone/push lifecycle above, the sandbox container applies its own restrictions to everything happening *inside* the clone:

- **Filesystem**: only the bind-mounted `run_dir` is writable; nothing on the host outside it is reachable.
- **Network**: egress restricted to an explicit allowlist (e.g. `pypi.org`, `registry.npmjs.org`) — NOT `github.com`, since push/clone happen outside the sandbox, via the orchestrator process, not from inside the container.
- **Resources**: memory and CPU limits (e.g. 512MB / 0.5 CPU) prevent a runaway process from affecting other concurrent runs on the same host.
- **Capabilities**: elevated syscalls dropped (`cap_drop: ALL`) — a compromised or buggy command inside the container cannot escalate privileges on the host.
- **Ephemeral by default**: the container itself is destroyed after each command (`remove=True`) — no state persists between individual shell calls beyond what's written to the mounted `run_dir`.

---

## 6. Failure-mode guarantees this design provides

| Scenario | What happens to the user's real repository |
|---|---|
| Verification fails | Untouched — Stage 4 never executes |
| Sandbox process crashes | Untouched — the crash is contained to the disposable clone/container |
| Server crashes mid-run (before push) | Untouched — an orphaned clone may need manual/scheduled cleanup, but no partial state ever reached GitHub |
| Server crashes mid-run (after push, before cleanup) | The push already succeeded; only cleanup of the local clone is delayed — the real repo update is intact either way |
| A generated script tries to reach an unapproved external host | Blocked at the network layer — never an implicit path to the real repo or any other external system |

The real repository can only ever end up in one of two states after a run: **exactly as it was before** (any failure prior to Stage 4), or **updated with a fully verified change** (successful completion of all five stages). There is no intermediate state where a partially-completed or unverified change reaches the user's actual project.
