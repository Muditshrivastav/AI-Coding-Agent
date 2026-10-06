# PM2 start script for AI Coding Agent API Server
# Run this once from the project root to launch the API in the background.
# Requirements: npm install -g pm2  (already installed)

param(
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$Logs,
    [switch]$Save
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step($msg) { Write-Host "`n>> $msg" -ForegroundColor Cyan }

if ($Stop) {
    Write-Step "Stopping api-server..."
    pm2 stop api-server
    exit 0
}

if ($Restart) {
    Write-Step "Restarting api-server..."
    pm2 restart ecosystem.config.js
    exit 0
}

if ($Status) {
    pm2 list
    exit 0
}

if ($Logs) {
    pm2 logs api-server --lines 100
    exit 0
}

# ── Default: start ───────────────────────────────────────────────────────────
Write-Step "Starting api-server with PM2..."

# Load .env so env vars (API keys etc.) are forwarded to the PM2 process
if (Test-Path ".env") {
    Write-Host "  Loading .env environment variables..." -ForegroundColor Gray
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $key   = $matches[1].Trim()
            $value = $matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "  Done." -ForegroundColor Gray
}

pm2 start ecosystem.config.js

Write-Step "API server is now running in the background."
Write-Host ""
Write-Host "  Health check : http://localhost:8000/health" -ForegroundColor Green
Write-Host "  Docs         : http://localhost:8000/docs"   -ForegroundColor Green
Write-Host ""
Write-Host "Useful PM2 commands:"
Write-Host "  pm2 logs api-server      -- tail live logs"
Write-Host "  pm2 monit                -- interactive dashboard"
Write-Host "  pm2 list                 -- show all processes"
Write-Host "  pm2 restart api-server   -- hot restart"
Write-Host "  pm2 stop api-server      -- stop the server"
Write-Host ""
Write-Host "Or use this script:"
Write-Host "  .\start-api.ps1 -Status"
Write-Host "  .\start-api.ps1 -Logs"
Write-Host "  .\start-api.ps1 -Restart"
Write-Host "  .\start-api.ps1 -Stop"
Write-Host ""

if ($Save) {
    Write-Step "Saving PM2 process list (survives reboots)..."
    pm2 save
}
