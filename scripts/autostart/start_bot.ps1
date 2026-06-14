. "$PSScriptRoot\common.ps1"

$repoRoot = Get-RepoRoot
Start-ManagedProcess `
    -Name "telegram_bot" `
    -WorkingDirectory $repoRoot `
    -ArgumentList @("-u", "-m", "bot.telegram_bot") `
    -StopPatterns @("bot.telegram_bot")

