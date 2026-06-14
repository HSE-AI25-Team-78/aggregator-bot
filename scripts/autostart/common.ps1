Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

function Get-LogDir {
    $logDir = Join-Path (Get-RepoRoot) "bot_data\runtime_logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    return $logDir
}

function Import-LocalEnv {
    $envPath = Join-Path (Get-RepoRoot) ".env"
    if (-not (Test-Path $envPath)) {
        return
    }
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim("'`"")
        if ($name) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Stop-ProcessesByPattern {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Patterns
    )

    $processes = Get-CimInstance Win32_Process | Where-Object {
        $process = $_
        $process.Name -in @("py.exe", "python.exe", "pwsh.exe", "powershell.exe") -and
        $null -ne $process.CommandLine -and
        ($Patterns | Where-Object { $_.Length -gt 0 -and $process.CommandLine -like "*$_*" })
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string[]]$StopPatterns
    )

    Import-LocalEnv
    $logDir = Get-LogDir
    $stdout = Join-Path $logDir "$Name.log"
    $stderr = Join-Path $logDir "$Name.err.log"
    Stop-ProcessesByPattern -Patterns $StopPatterns

    Start-Process `
        -FilePath "py" `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden
}
