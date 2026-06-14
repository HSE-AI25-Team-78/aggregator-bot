Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$pwsh = (Get-Command pwsh).Source
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
New-Item -ItemType Directory -Path $startupDir -Force | Out-Null

$launcherPath = Join-Path $startupDir "AggregatorBotSuite.cmd"
$scriptPath = Join-Path $repoRoot "scripts\autostart\start_all.ps1"

$cmdContent = @"
@echo off
start "" "$pwsh" -NoProfile -ExecutionPolicy Bypass -File "$scriptPath"
"@

Set-Content -Path $launcherPath -Value $cmdContent -Encoding ASCII

Write-Host "Startup launcher installed:"
Write-Host $launcherPath
Write-Host ""
Write-Host "It will start the full suite on user logon:"
Write-Host "- Telegram bot"
Write-Host "- Live dashboard"
Write-Host "- FastAPI service API"

