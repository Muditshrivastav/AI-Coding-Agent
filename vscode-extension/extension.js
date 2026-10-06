// @ts-check
'use strict';

/**
 * extension.js — AI Coding Agent VS Code Extension
 *
 * Responsibilities
 * ────────────────
 * 1. Register a WebviewViewProvider for the "codingAgent.ui" sidebar view.
 * 2. In resolveWebviewView:
 *      • enable scripts
 *      • set localResourceRoots to the built React dist/ folder
 *      • inject a strict Content-Security-Policy that allows
 *          - the webview's own nonce-tagged inline scripts
 *          - connections to http://127.0.0.1:8000 (local FastAPI)
 *          - Google Fonts CDN (used by the UI's stylesheet)
 *      • load the built index.html, rewriting every asset href/src with
 *        webview.asWebviewUri() so VS Code can serve local files
 * 3. Provide a "codingAgent.startApi" command that:
 *      • finds the repo's .venv Python
 *      • cd's to the currently open workspace folder
 *      • spawns  uvicorn interfaces.api_server:app  in a dedicated terminal
 * 4. Provide a "codingAgent.openUi" command that focuses the sidebar view.
 */

const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');

// ─── Constants ───────────────────────────────────────────────────────────────
const VIEW_ID = 'codingAgent.ui';
const API_BASE = 'http://127.0.0.1:8000';

// Track last known workspace so we can detect changes between polls
let _lastKnownWorkspace = '';

// Timestamp (ms) of the last API launch — used to give uvicorn time to boot
// before the poller declares the server "dead" and tries to restart it.
let _apiLastStartedAt = 0;

// How long (ms) to wait after a launch before health-check restarts are allowed.
const _API_BOOT_GRACE_MS = 45_000;

// Mutex: serialises concurrent calls to ensureApiRunning so only ONE terminal
// is ever started at a time (fixes the activate + resolveWebviewView race).
// When non-null, a start is already in progress and callers should await it.
/** @type {Promise<void> | null} */
let _startLock = null;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Return the absolute path to the built React app's index.html.
 * Looks in:
 * 1. Bundled dist inside extension: <extensionPath>/dist
 * 2. Development sibling path: <extensionPath>/../ui/dist
 * 3. Open workspace folders: <workspace>/ui/dist
 * 4. Known project path: D:\Machine Learning\AI Coding Agent\ui\dist
 */
function getDistDir(/** @type {vscode.ExtensionContext} */ ctx) {
  const candidates = [
    path.join(ctx.extensionPath, '..', 'ui', 'dist'),
    'd:\\Machine Learning\\AI Coding Agent\\ui\\dist',
    path.join(ctx.extensionPath, 'dist'),
    path.join(ctx.extensionPath, 'ui', 'dist'),
  ];

  const folders = vscode.workspace.workspaceFolders || [];
  for (const folder of folders) {
    candidates.push(path.join(folder.uri.fsPath, 'ui', 'dist'));
    candidates.push(path.join(folder.uri.fsPath, 'AI Coding Agent', 'ui', 'dist'));
  }

  // Find all candidate dist folders that have index.html and pick the newest one
  let newestDir = null;
  let newestTime = 0;

  for (const c of candidates) {
    const indexPath = path.join(c, 'index.html');
    if (fs.existsSync(indexPath)) {
      try {
        const mtime = fs.statSync(indexPath).mtimeMs;
        if (mtime > newestTime) {
          newestTime = mtime;
          newestDir = c;
        }
      } catch (_) {
        if (!newestDir) newestDir = c;
      }
    }
  }

  if (newestDir) {
    return newestDir;
  }

  // Fallback to default
  return path.join(ctx.extensionPath, '..', 'ui', 'dist');
}

/**
 * Generate a random nonce for the inline <script> CSP directive.
 */
function nonce() {
  return crypto.randomBytes(16).toString('base64');
}

/**
 * Read the built index.html and return it with every local asset path
 * ( href="./assets/…"  and  src="./assets/…" ) replaced by the VS Code
 * webview-safe URI produced by webview.asWebviewUri().
 *
 * Also injects the CSP <meta> tag and the nonce attribute.
 *
 * @param {vscode.Webview} webview
 * @param {string}         distDir
 * @param {string}         n        - nonce string
 * @returns {string}
 */
function buildHtml(webview, distDir, n) {
  const htmlPath = path.join(distDir, 'index.html');

  if (!fs.existsSync(htmlPath)) {
    // Guide the user to run the build first.
    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${n}';">
  <title>AI Coding Agent</title>
  <style>
    body{font-family:sans-serif;padding:20px;background:#121316;color:#f1f5f9}
    code{background:#222;padding:2px 6px;border-radius:4px}
    pre{background:#0d1117;padding:12px;border-radius:6px;overflow:auto;font-size:12px}
  </style>
</head>
<body>
  <h2>⚠️ React build not found</h2>
  <p>The extension expects a production build of the React UI at:</p>
  <pre><code>${htmlPath}</code></pre>
  <p>Run the following once from inside the <code>ui/</code> directory:</p>
  <pre><code>npm install && npm run build</code></pre>
  <p>Then reload the VS Code window (<kbd>Ctrl+Shift+P</kbd> → <em>Developer: Reload Window</em>).</p>
</body>
</html>`;
  }

  let html = fs.readFileSync(htmlPath, 'utf8');

  // ── Replace every  ./assets/<file>  reference with the webview URI ──────
  // Vite emits paths like  ./assets/index-AbCd.js  and  ./assets/index-XyZ.css
  // when base is set to './'
  html = html.replace(/(href|src)="(\.\/assets\/[^"]+)"/g, (_, attr, assetRel) => {
    const assetPath = path.join(distDir, assetRel.replace(/^\.\//, ''));
    const webviewUri = webview.asWebviewUri(vscode.Uri.file(assetPath));
    return `${attr}="${webviewUri}"`;
  });

  // ── Inject Content-Security-Policy ──────────────────────────────────────
  // Insert a <meta http-equiv="Content-Security-Policy"> right after <head>
  const csp = [
    `default-src 'none'`,
    // Scripts: only the built chunks (loaded via webview URI) + nonce for any
    // tiny inline bootstrap Vite may emit.
    `script-src 'nonce-${n}' ${webview.cspSource}`,
    // Styles: built chunks served from webview URI + inline styles the UI uses.
    `style-src 'unsafe-inline' ${webview.cspSource}`,
    // Fonts: Google Fonts CDN (Plus Jakarta Sans, JetBrains Mono)
    `font-src https://fonts.gstatic.com ${webview.cspSource}`,
    // Stylesheets from Google Fonts API
    `style-src-elem 'unsafe-inline' ${webview.cspSource} https://fonts.googleapis.com`,
    // API calls to local FastAPI server
    `connect-src ${API_BASE} ws://127.0.0.1:8000 http://localhost:8000 ws://localhost:8000`,
    // Allow data: URIs for image attachments
    `img-src ${webview.cspSource} data: blob:`,
  ].join('; ');

  const cspMeta = `<meta http-equiv="Content-Security-Policy" content="${csp}">`;
  html = html.replace(/<head>/i, `<head>\n  ${cspMeta}`);

  // ── Add nonce to any existing inline <script> tags Vite may emit ────────
  html = html.replace(/<script\b(?![^>]*\bnonce=)/gi, `<script nonce="${n}"`);

  return html;
}

// ─── WebviewViewProvider ──────────────────────────────────────────────────────

class CodingAgentViewProvider {
  /**
   * @param {vscode.ExtensionContext} ctx
   */
  constructor(ctx) {
    this._ctx = ctx;
    this._view = undefined;  // the live WebviewView, if resolved
  }

  /**
   * Called by VS Code when the sidebar panel becomes visible for the first time
   * (and each time it is "restored" after a window reload).
   *
   * @param {vscode.WebviewView}               webviewView
   * @param {vscode.WebviewViewResolveContext} _resolveContext
   * @param {vscode.CancellationToken}         _token
   */
  resolveWebviewView(webviewView, _resolveContext, _token) {
    this._view = webviewView;

    const distDir = getDistDir(this._ctx);

    // ── Webview options ──────────────────────────────────────────────────
    webviewView.webview.options = {
      // Allow the webview to run JavaScript.
      enableScripts: true,
      // Only allow loading resources from the built React dist/ folder.
      localResourceRoots: [
        vscode.Uri.file(distDir),
      ],
    };

    // ── Initial HTML ─────────────────────────────────────────────────────
    const n = nonce();
    webviewView.webview.html = buildHtml(webviewView.webview, distDir, n);

    // ── Automatically ensure the FastAPI backend is running whenever UI opens ──
    const repoRoot = resolveRepoRoot(this._ctx);
    ensureApiRunning(repoRoot, /* silent */ true);

    // ── Rebuild HTML when the view becomes visible again ─────────────────
    // (handles the case where the dist/ folder was updated while VS Code ran)
    webviewView.onDidChangeVisibility(() => {
      if (webviewView.visible) {
        const n2 = nonce();
        webviewView.webview.html = buildHtml(webviewView.webview, distDir, n2);
      }
    }, null, this._ctx.subscriptions);

    // ── Message bridge: webview → extension ─────────────────────────────
    // The React app can post messages for future extension-side actions.
    webviewView.webview.onDidReceiveMessage(
      (message) => {
        switch (message.command) {
          case 'startApi':
            vscode.commands.executeCommand('codingAgent.startApi');
            break;
          case 'alert':
            vscode.window.showInformationMessage(message.text);
            break;
        }
      },
      null,
      this._ctx.subscriptions,
    );
  }

  /**
   * Focus the sidebar view programmatically.
   */
  focus() {
    if (this._view) {
      this._view.show(true);
    }
  }
}

// ─── API Server launcher ──────────────────────────────────────────────────────

/**
 * @param {vscode.ExtensionContext} ctx
 * @returns {string}
 */
function resolveRepoRoot(ctx) {
  const candidate1 = path.resolve(ctx.extensionPath, '..');
  if (fs.existsSync(path.join(candidate1, 'interfaces', 'api_server.py'))) {
    return candidate1;
  }
  const knownProject = 'd:\\Machine Learning\\AI Coding Agent';
  if (fs.existsSync(path.join(knownProject, 'interfaces', 'api_server.py'))) {
    return knownProject;
  }
  const folders = vscode.workspace.workspaceFolders || [];
  for (const folder of folders) {
    const candidate = folder.uri.fsPath;
    if (fs.existsSync(path.join(candidate, 'interfaces', 'api_server.py'))) {
      return candidate;
    }
  }
  return candidate1;
}

/**
 * @param {string} repoRoot
 * @returns {string}
 */
function resolvePython(repoRoot) {
  // Windows: .venv\Scripts\python.exe
  const winPy = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  // Unix:    .venv/bin/python
  const unixPy = path.join(repoRoot, '.venv', 'bin', 'python');

  if (fs.existsSync(winPy)) return winPy;
  if (fs.existsSync(unixPy)) return unixPy;

  return process.platform === 'win32' ? 'python' : 'python3';
}

function getTargetWorkspace() {
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    return folders[0].uri.fsPath;
  }
  return null;
}

/**
 * Kill any existing 'Coding Agent API' terminal.
 */
function killApiTerminal() {
  const existing = vscode.window.terminals.find(t => t.name === 'Coding Agent API');
  if (existing) {
    existing.dispose();
  }
}

/**
 * Check whether the API server is currently responding on http://127.0.0.1:8000.
 * @returns {Promise<boolean>}
 */
function isApiAlive() {
  return new Promise((resolve) => {
    const http = require('http');
    const req = http.get('http://127.0.0.1:8000/health', { timeout: 2000 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

/**
 * Ask the running API what workspace root it currently has configured.
 * Returns null if the server is not reachable.
 * @returns {Promise<string|null>}
 */
function getApiWorkspace() {
  return new Promise((resolve) => {
    const http = require('http');
    let body = '';
    const req = http.get('http://127.0.0.1:8000/workspace', { timeout: 2000 }, (res) => {
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => {
        try { resolve(JSON.parse(body).workspace_root || null); }
        catch { resolve(null); }
      });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

/**
 * Ensure the API server is running without asking the user.
 *
 * Uses a promise-based mutex (_startLock) so that concurrent callers
 * (e.g. activate() and resolveWebviewView() both firing at startup) are
 * serialised — only the first caller actually launches uvicorn.
 *
 * @param {string} repoRoot
 * @param {boolean} silent
 */
async function ensureApiRunning(repoRoot, silent = true) {
  // If a start is already in progress, wait for it to finish and return.
  // After it finishes the server will be up, so we don't need to do anything.
  if (_startLock) {
    await _startLock;
    return;
  }

  // Fast-path: server is already alive — just sync workspace.
  const alive = await isApiAlive();
  if (alive) {
    const targetWorkspace = getTargetWorkspace() || repoRoot;
    if (targetWorkspace !== _lastKnownWorkspace) {
      _lastKnownWorkspace = targetWorkspace;
      notifyServerOfWorkspace(targetWorkspace, /* immediate */ true);
    }
    return;
  }

  // Acquire the lock — any concurrent caller that reaches here now will
  // await this promise and skip doing anything once it resolves.
  // releaseLock is initialised to a no-op so @ts-check knows it is always
  // callable, even though the Promise executor assigns it synchronously.
  /** @type {() => void} */
  let releaseLock = () => {};
  _startLock = new Promise(resolve => { releaseLock = resolve; });

  try {
    // Re-check liveness after acquiring the lock (a concurrent caller may
    // have just started the server while we were awaiting the lock).
    const aliveNow = await isApiAlive();
    if (aliveNow) {
      const targetWorkspace = getTargetWorkspace() || repoRoot;
      if (targetWorkspace !== _lastKnownWorkspace) {
        _lastKnownWorkspace = targetWorkspace;
        notifyServerOfWorkspace(targetWorkspace, /* immediate */ true);
      }
      return;
    }

    // Dispose any stale ghost terminal whose process has exited.
    const existing = vscode.window.terminals.find(t => t.name === 'Coding Agent API');
    if (existing) {
      console.log('[AI Coding Agent] Stale API terminal found (server not responding). Disposing and restarting...');
      existing.dispose();
      await new Promise(r => setTimeout(r, 500));
    }

    startApiInTerminal(repoRoot, false, silent);
  } finally {
    // Always release the lock so future callers are not blocked forever.
    _startLock = null;
    releaseLock();
  }
}

/**
 * Launch `uvicorn interfaces.api_server:app` using the repo's .venv Python.
 * Sets AGENT_ROOT_DIR to the active open workspace folder (e.g. google-adk).
 *
 * @param {string} repoRoot      - Absolute path to the AI Coding Agent repo (where api_server.py lives).
 * @param {boolean} forceRestart - If true, kill any existing API terminal before starting.
 * @param {boolean} silent       - If true, do not pop an info message.
 */
function startApiInTerminal(repoRoot, forceRestart = false, silent = false) {
  const python = resolvePython(repoRoot);
  const targetWorkspace = getTargetWorkspace() || repoRoot;

  // Check whether a terminal for the API is already running
  const existing = vscode.window.terminals.find(t => t.name === 'Coding Agent API');
  if (existing && !forceRestart) {
    if (!silent) {
      existing.show(false);
      vscode.window.showInformationMessage(
        'AI Coding Agent: API terminal is already open. Check the terminal panel at the bottom of VS Code.',
      );
    }
    return;
  }

  // Kill old terminal if restarting
  if (existing) {
    existing.dispose();
  }

  // Record start time so the poller won't kill uvicorn before it finishes booting
  _apiLastStartedAt = Date.now();

  const terminal = vscode.window.createTerminal({
    name: 'Coding Agent API',
    cwd: repoRoot,
    env: {
      AGENT_ROOT_DIR: targetWorkspace,
      // Explicitly blank VSCODE_CWD so it cannot override AGENT_ROOT_DIR
      // inside the Python process (VS Code sets it to the launch directory).
      VSCODE_CWD: '',
    },
  });
  terminal.show(false);

  // Switch to repo directory, set env vars, then start uvicorn in a restart loop.
  // --reload is intentionally omitted: harness/ file writes (AGENTS.md, sessions/)
  // trigger uvicorn restart loops. Instead we use a plain while-loop so the terminal
  // stays alive and uvicorn restarts automatically on crash — without --reload.
  terminal.sendText(`cd "${repoRoot}"`, true);
  terminal.sendText(`$env:AGENT_ROOT_DIR="${targetWorkspace}"`, true);
  terminal.sendText(`$env:VSCODE_CWD=""`, true);
  // PowerShell restart loop — keeps uvicorn running indefinitely without --reload.
  // Press Ctrl+C twice (once to stop uvicorn, once to break the loop) to exit manually.
  const cmd = [
    `while ($true) {`,
    `  Write-Host "[Coding Agent] Starting uvicorn (no --reload)..." -ForegroundColor Cyan`,
    `  & "${python}" -m uvicorn interfaces.api_server:app --host 127.0.0.1 --port 8000`,
    `  $exit = $LASTEXITCODE`,
    `  Write-Host "[Coding Agent] uvicorn exited (code $exit). Restarting in 2 s..." -ForegroundColor Yellow`,
    `  Start-Sleep -Seconds 2`,
    `}`,
  ].join('; ');
  terminal.sendText(cmd, true);

  // Track last known workspace
  _lastKnownWorkspace = targetWorkspace;

  // After uvicorn boots, push the correct workspace to the /workspace/set endpoint.
  // immediate=false means we wait 5 s before the first attempt (uvicorn needs time to start).
  notifyServerOfWorkspace(targetWorkspace, /* immediate */ false);

  if (!silent) {
    vscode.window.showInformationMessage(
      `AI Coding Agent: Starting API → workspace: ${vscode.workspace.name || targetWorkspace}`,
    );
  }
}

/**
 * Retry POST /workspace/set until the server accepts it.
 *
 * @param {string}  targetWorkspace
 * @param {boolean} immediate - true  → try right away (server already running);
 *                              false → wait 5 s first (server just launched).
 * @param {number}  maxRetries
 */
function notifyServerOfWorkspace(targetWorkspace, immediate = false, maxRetries = 30) {
  let attempts = 0;

  async function trySet() {
    attempts++;
    try {
      const http = require('http');
      const body = JSON.stringify({ root_dir: targetWorkspace });
      await new Promise((resolve, reject) => {
        const req = http.request(
          {
            hostname: '127.0.0.1', port: 8000, path: '/workspace/set', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
          },
          (res) => {
            res.resume();
            if (res.statusCode === 200) resolve(true);
            else reject(new Error(`HTTP ${res.statusCode}`));
          },
        );
        req.on('error', reject);
        req.write(body);
        req.end();
      });
      // Success — update tracking variable
      _lastKnownWorkspace = targetWorkspace;
      console.log(`[AI Coding Agent] Workspace set to: ${targetWorkspace}`);
    } catch (_err) {
      if (attempts < maxRetries) {
        setTimeout(trySet, 2000);
      } else {
        console.warn(`[AI Coding Agent] Could not set workspace after ${maxRetries} attempts.`);
      }
    }
  }

  if (immediate) {
    // Server is already up — attempt right away
    trySet();
  } else {
    // Server was just launched — give uvicorn ~5 s to finish booting
    setTimeout(trySet, 5000);
  }
}

// ─── Workspace-change poller ─────────────────────────────────────────────────

/**
 * Poll every 5 seconds.
 *
 * Detects two conditions automatically:
 *  1. The API server has crashed/exited → restart it.
 *  2. The open workspace folder changed (e.g. user hit "Open Folder") but
 *     onDidChangeWorkspaceFolders didn't fire → hot-swap via POST /workspace/set.
 *
 * @param {string} repoRoot
 * @returns {NodeJS.Timeout} handle — push a dispose wrapper into ctx.subscriptions
 */
function startWorkspacePoller(repoRoot) {
  // Guard flag: prevent overlapping restart attempts from the poller
  let _pollerRestarting = false;

  return setInterval(async () => {
    const currentWorkspace = getTargetWorkspace() || repoRoot;
    const alive = await isApiAlive();

    if (!alive) {
      // ── Boot-grace window ─────────────────────────────────────────────────
      // If uvicorn was launched recently, it may still be booting.
      // Do NOT kill/restart it — just wait for the next poll tick.
      const msSinceLaunch = Date.now() - _apiLastStartedAt;
      if (msSinceLaunch < _API_BOOT_GRACE_MS) {
        console.log(`[AI Coding Agent] Poller: server not yet up — waiting for boot (${Math.round(msSinceLaunch / 1000)}s / ${_API_BOOT_GRACE_MS / 1000}s grace).`);
        return;
      }

      // Grace period has passed and the server is still not responding.
      // Guard against concurrent restart attempts.
      if (_pollerRestarting) return;
      _pollerRestarting = true;

      try {
        // The 'Coding Agent API' terminal runs uvicorn inside a while-loop.
        // If the terminal is still alive, uvicorn is just mid-restart (2 s pause).
        // Do NOT dispose the terminal — let the loop bring it back on its own.
        const existing = vscode.window.terminals.find(t => t.name === 'Coding Agent API');
        if (existing) {
          console.log('[AI Coding Agent] Poller: API is down but terminal loop is alive — waiting for auto-restart...');
          return;
        }
        // Terminal is gone — spawn a fresh one with the restart loop.
        console.log('[AI Coding Agent] Poller: API terminal gone (grace period expired). Spawning new terminal...');
        startApiInTerminal(repoRoot, false, /* silent */ true);
      } finally {
        _pollerRestarting = false;
      }
      return;
    }

    // Server alive — sync workspace if it changed
    if (currentWorkspace !== _lastKnownWorkspace) {
      console.log(`[AI Coding Agent] Poller: workspace changed ${_lastKnownWorkspace} → ${currentWorkspace}`);
      _lastKnownWorkspace = currentWorkspace;
      notifyServerOfWorkspace(currentWorkspace, /* immediate */ true);
    }
  }, 5000);
}

// ─── Extension entry points ──────────────────────────────────────────────────

/**
 * @param {vscode.ExtensionContext} ctx
 */
function activate(ctx) {
  const repoRoot = resolveRepoRoot(ctx);

  // ── Initialise last-known workspace tracking ──────────────────────────────
  _lastKnownWorkspace = getTargetWorkspace() || repoRoot;

  // ── Auto-start DISABLED ────────────────────────────────────────────────────
  // The extension no longer spawns the API terminal automatically on activation,
  // on sidebar open, or via the background health-check poller.
  //
  // To start the API server use ONE of:
  //   • PM2 (recommended, stays alive across terminal sessions):
  //       pm2 start ecosystem.config.js
  //       .\start-api.ps1
  //   • Manual one-off (dies when terminal closes):
  //       Ctrl+Shift+P → "AI Coding Agent: Start API"

  // ── Register the WebviewViewProvider ────────────────────────────────────
  const provider = new CodingAgentViewProvider(ctx);
  ctx.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_ID, provider, {
      // Keep the webview alive even when the panel is hidden so the React
      // component doesn't lose its in-memory state.
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  // ── Command: start the FastAPI server (manual, on-demand only) ───────────
  ctx.subscriptions.push(
    vscode.commands.registerCommand('codingAgent.startApi', () => {
      startApiInTerminal(repoRoot, /* forceRestart */ true);
    }),
  );

  // ── Command: focus / reveal the sidebar UI ───────────────────────────────
  ctx.subscriptions.push(
    vscode.commands.registerCommand('codingAgent.openUi', () => {
      provider.focus();
      // VS Code built-in command to open and focus the view container
      vscode.commands.executeCommand(`${VIEW_ID}.focus`);
    }),
  );

  // ── Workspace folder change handler ─────────────────────────────────────
  // When the workspace changes, update the API workspace (only if it's already
  // running — no auto-restart).
  ctx.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(() => {
      const newWorkspace = getTargetWorkspace();
      if (!newWorkspace || newWorkspace === _lastKnownWorkspace) return;

      console.log(`[AI Coding Agent] onDidChangeWorkspaceFolders: ${_lastKnownWorkspace} → ${newWorkspace}`);
      _lastKnownWorkspace = newWorkspace;

      // Hot-swap if server is alive, otherwise restart
      isApiAlive().then(alive => {
        if (alive) {
          notifyServerOfWorkspace(newWorkspace, /* immediate */ true);
        }
        // If server is not alive, do nothing — user will start it via PM2 or command.
      });
    }),
  );

  // ── Periodic health + workspace poller (every 5 s) ───────────────────────
  // Automatically restarts a crashed API and detects workspace folder changes
  // that did not fire onDidChangeWorkspaceFolders (e.g. "Open Folder" dialog).
  const pollerHandle = startWorkspacePoller(repoRoot);
  ctx.subscriptions.push({ dispose: () => clearInterval(pollerHandle) });
}

function deactivate() {
  // Nothing to clean up — VS Code disposes subscriptions automatically.
}

module.exports = { activate, deactivate };
