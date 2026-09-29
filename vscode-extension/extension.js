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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Return the absolute path to the built React app's index.html.
 * The dist/ folder lives at  <extension-root>/../ui/dist/  so the extension
 * doesn't need to bundle the React code itself.
 */
function getDistDir(/** @type {vscode.ExtensionContext} */ ctx) {
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
 * Resolve the Python interpreter inside the repo's .venv.
 * Falls back to the system "python" / "python3" if .venv doesn't exist.
 *
 * @param {string} repoRoot — absolute path to the repository root
 * @returns {string}
 */
function resolvePython(repoRoot) {
  // Windows: .venv\Scripts\python.exe
  const winPy = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  // Unix:    .venv/bin/python
  const unixPy = path.join(repoRoot, '.venv', 'bin', 'python');

  if (fs.existsSync(winPy)) return winPy;
  if (fs.existsSync(unixPy)) return unixPy;

  // Fallback — let the shell find it
  return process.platform === 'win32' ? 'python' : 'python3';
}

/**
 * Launch `uvicorn interfaces.api_server:app` in the workspace folder using
 * the repo's .venv Python (via `python -m uvicorn`).
 *
 * @param {string} repoRoot
 */
function startApiInTerminal(repoRoot) {
  const python = resolvePython(repoRoot);

  // Check whether a terminal for the API is already running
  const existing = vscode.window.terminals.find(t => t.name === 'Coding Agent API');
  if (existing) {
    existing.show(false);
    vscode.window.showInformationMessage(
      'AI Coding Agent: API terminal is already open. If the server stopped, type: python -m uvicorn interfaces.api_server:app --port 8000 --reload',
    );
    return;
  }

  const terminal = vscode.window.createTerminal({
    name: 'Coding Agent API',
    cwd: repoRoot,
  });
  terminal.show(false);

  // Use & "path/to/python.exe" syntax which works cleanly across PowerShell and CMD
  const cmd = `& "${python}" -m uvicorn interfaces.api_server:app --host 127.0.0.1 --port 8000 --reload`;
  terminal.sendText(cmd, true);

  vscode.window.showInformationMessage(
    `AI Coding Agent: API server starting on ${API_BASE} …`,
  );
}

// ─── Extension entry points ──────────────────────────────────────────────────

/**
 * @param {vscode.ExtensionContext} ctx
 */
function activate(ctx) {
  // Determine the repo root: prefer the first workspace folder that contains
  // the known .venv directory; fall back to extensionPath/../
  let repoRoot = path.join(ctx.extensionPath, '..');
  const folders = vscode.workspace.workspaceFolders;
  if (folders && folders.length > 0) {
    // Prefer the workspace folder that actually contains our repo
    for (const folder of folders) {
      const candidate = folder.uri.fsPath;
      if (
        fs.existsSync(path.join(candidate, '.venv')) ||
        fs.existsSync(path.join(candidate, 'interfaces', 'api_server.py'))
      ) {
        repoRoot = candidate;
        break;
      }
    }
    // If none matched specifically, default to first folder
    if (repoRoot === path.join(ctx.extensionPath, '..')) {
      repoRoot = folders[0].uri.fsPath;
    }
  }

  // ── Register the WebviewViewProvider ────────────────────────────────────
  const provider = new CodingAgentViewProvider(ctx);
  ctx.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_ID, provider, {
      // Keep the webview alive even when the panel is hidden so the React
      // component doesn't lose its in-memory state.
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  // ── Command: start the FastAPI server ────────────────────────────────────
  ctx.subscriptions.push(
    vscode.commands.registerCommand('codingAgent.startApi', () => {
      startApiInTerminal(repoRoot);
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

  // Auto-start convenience: notify the user they can launch the API
  vscode.window.showInformationMessage(
    'AI Coding Agent extension activated. Use the sidebar icon or run "AI Coding Agent: Start API Server".',
    'Start API',
  ).then(choice => {
    if (choice === 'Start API') {
      startApiInTerminal(repoRoot);
    }
  });
}

function deactivate() {
  // Nothing to clean up — VS Code disposes subscriptions automatically.
}

module.exports = { activate, deactivate };
