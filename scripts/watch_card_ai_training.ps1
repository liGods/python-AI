param(
    [int]$TargetGames = 200000,
    [int]$RefreshSeconds = 2,
    [int]$Workers = 12
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$trainingRoot = Join-Path $projectRoot "data\card_ai\training"
$statePath = Join-Path $trainingRoot "continuous_state.json"
$logDirectory = Join-Path $trainingRoot "logs"
$selfPlayRoot = Join-Path $trainingRoot "self_play"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python was not found: $pythonPath"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

function Get-TrainingProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*ok_tasks.card_ai continuous*" -and
        $_.CommandLine -like "*--target-games $TargetGames*"
    })
}

function Start-CardAiTraining {
    $env:CARD_AI_INFERENCE_BACKEND = "cuda"
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $logDirectory "continuous_${TargetGames}_$stamp.out.log"
    $stderrPath = Join-Path $logDirectory "continuous_${TargetGames}_$stamp.err.log"
    $arguments = @(
        "-m", "ok_tasks.card_ai", "continuous",
        "--project-root", $projectRoot,
        "--target-games", "$TargetGames",
        "--batch-games", "10000",
        "--train-every", "$TargetGames",
        "--evaluation-deals", "50000",
        "--workers", "$Workers"
    )
    Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden | Out-Null
}

function Read-TrainingState {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return $null
    }
    try {
        Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    }
    catch {
        $null
    }
}

function Get-CurrentProgress($state) {
    $committed = if ($null -ne $state) { [int]$state.completed_games } else { 0 }
    $cycle = if ($null -ne $state) { [int]$state.cycle + 1 } else { 1 }
    $cycleName = "cycle_{0:D5}" -f $cycle
    $trajectoryDirectory = Join-Path (Join-Path $selfPlayRoot $cycleName) "trajectories"
    $partial = 0
    if (Test-Path -LiteralPath $trajectoryDirectory) {
        $partial = @([IO.Directory]::EnumerateFiles($trajectoryDirectory, "*.jsonl.gz")).Count
    }
    [pscustomobject]@{
        Committed = $committed
        Partial = $partial
        Total = [Math]::Min($TargetGames, $committed + $partial)
        Cycle = $cycleName
    }
}

$lastTotal = 0
$lastSampleTime = Get-Date
$smoothedRate = 0.0
$restartCount = 0

while ($true) {
    $state = Read-TrainingState
    $progress = Get-CurrentProgress $state
    $processes = Get-TrainingProcesses
    $status = if ($null -ne $state) { [string]$state.status } else { "starting" }

    $terminalStatuses = @("completed", "plateau", "paused_disk", "failed")
    if ($processes.Count -eq 0 -and $status -notin $terminalStatuses) {
        Start-CardAiTraining
        $restartCount++
        Start-Sleep -Seconds 3
        $processes = Get-TrainingProcesses
    }

    $now = Get-Date
    $elapsed = [Math]::Max(0.1, ($now - $lastSampleTime).TotalSeconds)
    $instantRate = [Math]::Max(0, ($progress.Total - $lastTotal) / $elapsed)
    if ($instantRate -gt 0) {
        $smoothedRate = if ($smoothedRate -eq 0) { $instantRate } else { $smoothedRate * 0.8 + $instantRate * 0.2 }
    }
    $remaining = [Math]::Max(0, $TargetGames - $progress.Total)
    $eta = if ($smoothedRate -gt 0) { [TimeSpan]::FromSeconds($remaining / $smoothedRate) } else { $null }
    $percent = if ($TargetGames -gt 0) { $progress.Total * 100.0 / $TargetGames } else { 0 }
    $latestError = Get-ChildItem -LiteralPath $logDirectory -Filter "*.err.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $errorBytes = if ($null -ne $latestError) { $latestError.Length } else { 0 }

    Clear-Host
    Write-Host "BaiJiangPai AI - Live Training Progress" -ForegroundColor Cyan
    Write-Host ("Updated:          {0:yyyy-MM-dd HH:mm:ss}" -f $now)
    Write-Host ("Status:           {0}" -f $status)
    Write-Host ("Process:          {0}" -f $(if ($processes.Count -gt 0) { "RUNNING" } else { "STOPPED" })) -ForegroundColor $(if ($processes.Count -gt 0) { "Green" } else { "Red" })
    Write-Host ("Progress:         {0:N0} / {1:N0} ({2:N2}%)" -f $progress.Total, $TargetGames, $percent) -ForegroundColor Yellow
    Write-Host ("Committed:        {0:N0}" -f $progress.Committed)
    Write-Host ("Current batch:    {0:N0} ({1})" -f $progress.Partial, $progress.Cycle)
    Write-Host ("Recent speed:     {0:N1} games/sec" -f $smoothedRate)
    Write-Host ("ETA:              {0}" -f $(if ($null -ne $eta) { $eta.ToString("dd\.hh\:mm\:ss") } else { "calculating" }))
    Write-Host ("Auto restarts:    {0}" -f $restartCount)
    Write-Host ("Latest error log: {0} bytes" -f $errorBytes)
    Write-Host ""
    Write-Host "Keep this window open. Press Ctrl+C to stop monitoring." -ForegroundColor DarkGray

    if ($progress.Total -ge $TargetGames -or $status -eq "completed") {
        Write-Host "Target reached. Model training/evaluation may still be finishing; check the status line." -ForegroundColor Green
    }

    if ($processes.Count -eq 0 -and $status -in $terminalStatuses) {
        break
    }

    $lastTotal = $progress.Total
    $lastSampleTime = $now
    Start-Sleep -Seconds $RefreshSeconds
}
