/**
 * ecosystem.config.js — PM2 process configuration for the AI Coding Agent API.
 *
 * Usage:
 *   pm2 start ecosystem.config.js          # start all processes
 *   pm2 stop ecosystem.config.js           # stop all
 *   pm2 restart ecosystem.config.js        # restart all
 *   pm2 logs api-server                    # tail live logs
 *   pm2 monit                              # interactive dashboard
 *   pm2 save                               # persist process list across reboots
 *   pm2 startup                            # generate OS-level auto-start hook
 */

const path = require("path");

// Resolve the project root relative to this config file
const PROJECT_ROOT = __dirname;

// Detect the Python / uv executable to use
// Priority: uv venv > system python
const PYTHON = path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe");

module.exports = {
  apps: [
    // ── 1. FastAPI / uvicorn — main HTTP + SSE backend ─────────────────────
    {
      name: "api-server",
      interpreter: PYTHON,
      script: "-m",
      // NOTE: --reload is intentionally OMITTED.
      // The API writes files into harness/ (AGENTS.md, sessions/*.json) on every
      // request. With --reload active, uvicorn detects those writes and restarts
      // itself, creating an infinite start→write→reload→start loop.
      // For hot-reload during development use:  pm2 restart api-server
      // or add:  --reload-exclude 'harness/*' --reload-exclude 'logs/*'
      args: "uvicorn interfaces.api_server:app --host 127.0.0.1 --port 8000",
      cwd: PROJECT_ROOT,

      // Restart policy
      autorestart: true,
      watch: false,          // PM2 file-watching also disabled (harness/ writes would trigger loops)
      max_restarts: 10,      // stop thrashing if it keeps crashing
      min_uptime: "5s",      // a process must stay up >= 5 s to count as "stable"
      restart_delay: 2000,   // wait 2 s between restarts

      // Resource limits (optional safety net)
      max_memory_restart: "1G",

      // Logging — written to ./logs/ relative to cwd
      out_file: path.join(PROJECT_ROOT, "logs", "api-server.out.log"),
      error_file: path.join(PROJECT_ROOT, "logs", "api-server.err.log"),
      merge_logs: false,
      log_date_format: "YYYY-MM-DD HH:mm:ss",

      // Environment variables — merged on top of the shell env
      env: {
        NODE_ENV: "development",
        PYTHONUNBUFFERED: "1",      // flush stdout immediately (no line-buffering)
        PYTHONDONTWRITEBYTECODE: "1",
      },
      env_production: {
        NODE_ENV: "production",
        PYTHONUNBUFFERED: "1",
        PYTHONDONTWRITEBYTECODE: "1",
      },
    },

    // ── 2. ACP / stdio server — VS Code extension integration ──────────────
    // Uncomment if you want PM2 to manage the ACP server as well.
    // {
    //   name: "acp-server",
    //   interpreter: PYTHON,
    //   script: path.join(PROJECT_ROOT, "interfaces", "acp_server.py"),
    //   cwd: PROJECT_ROOT,
    //   autorestart: false,   // ACP is usually invoked by the VS Code extension
    //   out_file: path.join(PROJECT_ROOT, "logs", "acp-server.out.log"),
    //   error_file: path.join(PROJECT_ROOT, "logs", "acp-server.err.log"),
    //   env: { PYTHONUNBUFFERED: "1" },
    // },
  ],
};
