. "$PSScriptRoot\common.ps1"

Stop-ProcessesByPattern -Patterns @(
    "bot.telegram_bot",
    "bot.live_dashboard",
    "uvicorn app:app",
    "service.app"
)

