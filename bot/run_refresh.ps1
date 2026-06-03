$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $repoRoot "bot_data"
$stdoutPath = Join-Path $logDir "refresh_cron.log"
$stderrPath = Join-Path $logDir "refresh_cron.err.log"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Starting refresh pipeline" | Out-File -FilePath $stdoutPath -Encoding utf8 -Append

try {
    $output = py -m bot.refresh_pipeline 2>&1
    if ($output) {
        $output | Out-File -FilePath $stdoutPath -Encoding utf8 -Append
    }
    $finishedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$finishedAt] Refresh pipeline finished successfully" | Out-File -FilePath $stdoutPath -Encoding utf8 -Append
}
catch {
    $failedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$failedAt] Refresh pipeline failed: $($_.Exception.Message)" | Out-File -FilePath $stderrPath -Encoding utf8 -Append
    throw
}
