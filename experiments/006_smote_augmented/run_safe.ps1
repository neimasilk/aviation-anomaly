# Safe runner for Experiment 006
# This script avoids Tee-Object buffer overflow issues by using Python's native logging

param(
    [string]$LogFile = "../../../logs/006_training_safe.log"
)

# Create log directory if not exists
$LogDir = Split-Path -Parent $LogFile
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "Starting Experiment 006 with safe logging..." -ForegroundColor Cyan
Write-Host "Log file: $LogFile" -ForegroundColor Gray

# Set environment variables for better error handling
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_LAUNCH_BLOCKING = "0"  # Set to "1" only for debugging

# Run Python script with output redirect to file
# Using Start-Process to properly capture exit code
$process = Start-Process -FilePath "python" -ArgumentList "run.py" -NoNewWindow -PassThru -Wait -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.error"

$exitCode = $process.ExitCode

if ($exitCode -eq 0) {
    Write-Host "Experiment completed successfully!" -ForegroundColor Green
} elseif ($exitCode -eq -1073740791) {
    Write-Host "ERROR: CUDA Out of Memory!" -ForegroundColor Red
    Write-Host "Try reducing batch_size in config.yaml" -ForegroundColor Yellow
} elseif ($exitCode -eq -1073741819) {
    Write-Host "ERROR: Access Violation (possible memory corruption)!" -ForegroundColor Red
} else {
    Write-Host "Experiment failed with exit code: $exitCode" -ForegroundColor Red
}

# Display last 50 lines of log
Write-Host "`nLast 50 lines of log:" -ForegroundColor Cyan
Get-Content $LogFile -Tail 50 -ErrorAction SilentlyContinue

exit $exitCode
