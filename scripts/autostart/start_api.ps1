. "$PSScriptRoot\common.ps1"

$repoRoot = Get-RepoRoot
$serviceDir = Join-Path $repoRoot "service"
Start-ManagedProcess `
    -Name "service_api" `
    -WorkingDirectory $serviceDir `
    -ArgumentList @("-u", "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000") `
    -StopPatterns @("uvicorn app:app", "service.app", "127.0.0.1 8000")

