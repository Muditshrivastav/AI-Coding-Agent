# Implementation Plan: FastAPI Endpoints for Coding Agent

## Overview
This plan describes the steps required to add a **FastAPI‑based HTTP interface** to the existing autonomous coding harness.  The new service will expose endpoints that accept coding tasks (e.g., “write a function”, “refactor this file”, “run tests”), delegate the work to the existing harness, and return structured results.

---

## 1. Scope
- **In‑scope**
  - Set up a FastAPI project (virtual environment, dependencies).
  - Define request/response models with Pydantic.
  - Implement API routes for submitting a coding task, checking status, and retrieving results.
  - Wire the routes to the existing *coding‑harness* library (the `Agent` class and its helper functions).
  - Add unit‑ and integration‑tests covering the API layer.
  - Provide a Dockerfile and basic CI/CD steps for deployment.
- **Out‑of‑scope**
  - Re‑architecting the autonomous harness itself.
  - Adding a front‑end UI (outside of the pure API contract).
  - Long‑term scaling (Kubernetes, autoscaling, etc.) – only a basic containerised deployment is covered.

---

## 2. Requirements
| Category | Requirement |
|---|---|
| **Functional** | • `POST /tasks` – submit a coding task and receive a task identifier. |
| | • `GET /tasks/{task_id}` – poll for task status (`queued`, `running`, `completed`, `failed`). |
| | • `GET /tasks/{task_id}/result` – retrieve the final output (generated code, logs, diagnostics). |
| | • Optional `GET /health` endpoint for liveness/readiness checks. |
| **Non‑functional** | • Input validation using Pydantic models.
| | • Asynchronous processing – tasks are handed off to a background worker (e.g., `BackgroundTasks` or a simple in‑memory queue) so the API returns immediately.
| | • Minimal latency for status queries (≤ 200 ms).
| | • Error handling with clear HTTP status codes (400, 404, 500). |
| | • Logging of request/response metadata.
| **Security** | • Basic API‑key authentication (configurable via environment variable). |
| | • Rate‑limit per API key (optional, can be added later). |
| **Compatibility** | • Must run on Python 3.10+.
| | • Should not alter the public interface of the existing harness. |

---

## 3. Architectural Constraints
- **Dependency Isolation** – The FastAPI service lives in its own package (`coding_agent_api/`) and imports the existing harness as a library (`from coding_harness import Agent`).
- **Statelessness** – API itself is stateless; task state is stored in a lightweight in‑memory store (`dict`) for the MVP.  A future implementation can swap this for Redis or a DB without changing the endpoint signatures.
- **Asynchronous Execution** – FastAPI’s `BackgroundTasks` will be used to run the harness asynchronously, keeping the request thread free.
- **Single‑process** – For simplicity, the initial version runs a single Uvicorn worker.  Documentation will note how to scale with multiple workers and a shared task store.
- **Dockerised** – The service must be runnable via a Docker container exposing port `8000`.

---

## 4. Acceptance Criteria
1. **All endpoints are reachable** and return the expected JSON payloads.
2. **Submitting a task** returns a UUID and the task appears in the internal store with status `queued` → `running` → `completed`.
3. **Result endpoint** returns the generated code, any logs, and a success flag.  Errors from the harness are propagated as `failed` with an error message.
4. **Automated tests** achieve ≥ 90 % coverage for the API layer.
5. **Docker image builds** without errors and the container starts with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
6. **Documentation** – Swagger UI (automatically provided by FastAPI) shows correct schemas; a `README.md` with build/run instructions is present.

---

## 5. File List (Project Layout)
```
project_root/
├─ coding_agent_api/                # FastAPI service package
│   ├─ __init__.py
│   ├─ main.py                     # FastAPI app creation & entrypoint
│   ├─ router.py                   # API route definitions
│   ├─ schemas.py                  # Pydantic request/response models
│   ├─ tasks.py                    # In‑memory task store & background logic
│   └─ dependencies.py             # API‑key auth dependency
│
├─ tests/                           # Test suite
│   ├─ __init__.py
│   ├─ conftest.py                 # Pytest fixtures (client, mock harness)
│   ├─ test_router.py              # Endpoint tests (status codes, payloads)
│   └─ test_tasks.py                # Task handling logic tests
│
├─ Dockerfile                       # Container build definition
├─ requirements.txt                  # Pin fastapi, uvicorn, pydantic, etc.
├─ .env.example                     # Example env vars (API_KEY, LOG_LEVEL)
├─ README.md                        # Project overview & run instructions
└─ PLAN.md                          # THIS DOCUMENT
```

---

## 6. Detailed Steps
### 6.1 Project Setup
1. **Create virtual environment** (`python -m venv .venv`).
2. **Add dependencies** to `requirements.txt`:
   ```text
   fastapi>=0.109.0
   uvicorn[standard]>=0.27.0
   pydantic>=2.5.0
   python-dotenv>=1.0.0   # for loading .env files
   ```
   *If the harness has its own dependencies, add them to the same file.*
3. **Initialize git repo** and commit the skeleton files.
4. **Create `coding_agent_api/` package** with the files listed above.

---

### 6.2 API Schemas (`schemas.py`)
- `TaskRequest` – fields: `prompt: str`, optional `context: dict` (e.g., existing files), `language: str` (default "python").
- `TaskResponse` – fields: `task_id: str`, `status: Literal["queued","running","completed","failed"]`.
- `TaskResult` – fields: `task_id`, `success: bool`, `generated_code: Optional[str]`, `log: str`, `error: Optional[str]`.
- `HealthResponse` – simple `status: str`.

---

### 6.3 In‑Memory Task Store (`tasks.py`)
- Use a global `dict[UUID, TaskInfo]` where `TaskInfo` is a dataclass capturing `status`, `result`, timestamps.
- Provide functions:
  - `create_task(request: TaskRequest) -> UUID`
  - `update_status(uuid, status)`
  - `set_result(uuid, result)`
  - `get_status(uuid)`
  - `get_result(uuid)`
- Background worker function `run_task(uuid, request)` that:
  1. Marks status `running`.
  2. Calls the existing harness: `Agent().handle(request.prompt, ...)` (or whichever entry point exists).
  3. Captures output, populates `TaskResult`, sets status `completed` or `failed`.
- Use FastAPI's `BackgroundTasks` to schedule `run_task`.

---

### 6.4 Routing (`router.py`)
```python
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
router = APIRouter()

@router.post("/tasks", response_model=TaskResponse)
async def submit_task(request: TaskRequest, background: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    task_id = create_task(request)
    background.add_task(run_task, task_id, request)
    return TaskResponse(task_id=task_id, status="queued")

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_status(task_id: str, api_key: str = Depends(verify_api_key)):
    info = get_status(task_id)
    if not info:
        raise HTTPException(404, "Task not found")
    return TaskResponse(task_id=task_id, status=info.status)

@router.get("/tasks/{task_id}/result", response_model=TaskResult)
async def get_result(task_id: str, api_key: str = Depends(verify_api_key)):
    result = get_result(task_id)
    if not result:
        raise HTTPException(404, "Result not available")
    return result

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")
```
(Actual code will be placed in `router.py`.)

---

### 6.5 Application Entry Point (`main.py`)
- Create FastAPI instance, include router, add middleware for logging.
- Load environment variables (`API_KEY`, `LOG_LEVEL`).
- Optionally add CORS middleware if external clients are expected.

---

### 6.6 Authentication Dependency (`dependencies.py`)
- Simple API‑key check:
  ```python
  from fastapi import Header, HTTPException, Security
  def verify_api_key(x_api_key: str = Header(...)):
      expected = os.getenv("API_KEY")
      if x_api_key != expected:
          raise HTTPException(status_code=401, detail="Invalid API key")
      return x_api_key
  ```
- Documentation notes that the header name is `X-API-Key`.

---

### 6.7 Integration with Existing Harness
- Identify the public entry point of the autonomous coding harness (e.g., a class `CodingAgent` with a method `run(prompt, context)`).
- Add a thin wrapper in `tasks.py` that imports this class and forwards the request.
- Ensure the harness runs synchronously inside the background task; if it already spawns its own threads/processes, simply await/collect the result.
- Keep the wrapper small so future changes to the harness API only affect a single location.

---

### 6.8 Testing Strategy
1. **Unit Tests** (`tests/test_router.py`)
   - Use `TestClient` from FastAPI.
   - Mock the harness (e.g., with `unittest.mock`) to return a deterministic result.
   - Verify status codes, response bodies, and that `BackgroundTasks` is called.
2. **Task Logic Tests** (`tests/test_tasks.py`)
   - Test the in‑memory store functions (create, update, retrieve).
   - Simulate a task run using a dummy harness implementation.
3. **Integration Test** (optional)
   - Spin up the app with `uvicorn` in a subprocess and send real HTTP requests.
4. **Coverage** – Add `pytest-cov` to `requirements.txt` and enforce 90 % coverage on the `coding_agent_api/` package.

---

### 6.9 Dockerisation
- **Dockerfile** (multi‑stage):
  ```dockerfile
  FROM python:3.11-slim AS builder
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --upgrade pip && pip install --user -r requirements.txt

  FROM python:3.11-slim
  WORKDIR /app
  COPY --from=builder /root/.local /root/.local
  ENV PATH=/root/.local/bin:$PATH
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "coding_agent_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- **`.dockerignore`** to exclude `__pycache__`, `.git`, `.venv`, etc.
- Provide a `docker-compose.yml` example for local development (optional).

---

### 6.10 CI/CD (GitHub Actions example)
- **Workflow** `ci.yml`:
  1. Checkout code.
  2. Set up Python 3.11.
  3. Install dependencies.
  4. Run `pytest --cov=coding_agent_api`.
  5. Build Docker image and optionally push to a registry on `main`.
- Linting with `ruff` or `flake8` can be added.

---

## 7. Documentation & Deliverables
- **README.md** – Quick‑start guide (setup, run, test, docker).  Include example `curl` commands.
- **OpenAPI spec** – automatically generated and accessible at `/docs` and `/openapi.json`.
- **PLAN.md** – this file, placed at repository root.

---

## 8. Risks & Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| In‑memory task store loses data on restart | Low (MVP) | Document that persistence is future work; optionally add a flag to persist to a JSON file for dev.
| Harness raises unexpected exceptions | Medium | Wrap calls in a generic `try/except` block, capture stack trace in `TaskResult.error` and set status `failed`.
| API‑key leakage | High | Encourage use of environment variables and Docker secrets; do not hard‑code defaults.
| Blocking long‑running tasks overload workers | Medium | Use `BackgroundTasks` with a configurable thread pool (`max_workers`). For heavy loads, replace with a proper queue system (Celery, RQ).

---

## 9. Timeline (Suggested)
| Day | Activity |
|---|---|
| 1 | Project scaffolding, dependencies, CI pipeline set up |
| 2 | Implement schemas, task store, and background runner |
| 3 | Add routers and authentication dependency |
| 4 | Write unit & integration tests, achieve coverage target |
| 5 | Dockerfile, readme, and local Docker testing |
| 6 | Code review, final adjustments, merge |
| 7 | Release candidate build and optional deployment to a staging environment |

---

*End of Plan*