# Quick script to run commands in the virtual environment
# Usage: .\run_in_venv.ps1 "python src\train.py"

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

$VENV_NAME = "text2sql_env"
$VENV_PATH = ".\$VENV_NAME"

# Check if venv exists
if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\setup_venv.ps1" -ForegroundColor Yellow
    exit 1
}

# Activate and run command
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& "$VENV_PATH\Scripts\Activate.ps1"

Write-Host "Running: $Command" -ForegroundColor Green
Write-Host ""
Invoke-Expression $Command

# Note: Deactivation happens automatically when script ends
