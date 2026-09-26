# Antigravity / Codex AI Coding Agent UI

A web frontend for the Autonomous Coding Agent Harness modeled after **Google Antigravity & OpenAI Codex**.

## Features

- **Antigravity Glassmorphism & Dark Aesthetic**: Rich HSL gradients, cyan/purple glow tokens, and responsive layout.
- **Multi-Session Management**: Create, switch, and delete multiple agent execution threads via `/sessions`.
- **Human-in-the-Loop (HITL) Guard Gate**: Real-time interactive approval cards with `Approve & Proceed` or `Reject` buttons when `permissions.json` triggers an `ask` rule.
- **Workspace Explorer & Code Viewer**: Inspect workspace files and view code directly in the side panel.
- **Local Sandbox Mode**: Zero Docker requirement—directly runs commands via local shell with HarnessGuard interception.

## Quickstart

### 1. Start the Backend API
In the root directory:
```bash
uvicorn interfaces.api_server:app --reload --port 8000
```

### 2. Start the React Frontend
In the `ui` directory:
```bash
cd ui
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.
