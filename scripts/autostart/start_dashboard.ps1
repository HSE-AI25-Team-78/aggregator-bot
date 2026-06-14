. "$PSScriptRoot\common.ps1"

$repoRoot = Get-RepoRoot
Start-ManagedProcess `
    -Name "live_dashboard" `
    -WorkingDirectory $repoRoot `
    -ArgumentList @("-u", "-m", "bot.live_dashboard") `
    -StopPatterns @("bot.live_dashboard")

