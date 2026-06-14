. "$PSScriptRoot\common.ps1"

$repoRoot = Get-RepoRoot
$dataDir = Join-Path $repoRoot "data"
$existingCorpus = @(
    Get-ChildItem -Path $dataDir -Filter *.csv -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "all_channels_combined.csv" }
)

if ($existingCorpus.Count -eq 0) {
    Write-Host "Corpus is empty. Bootstrapping base channels via refresh pipeline..."
    & py -m bot.refresh_pipeline --force
}

& "$PSScriptRoot\start_bot.ps1"
& "$PSScriptRoot\start_dashboard.ps1"
& "$PSScriptRoot\start_api.ps1"
