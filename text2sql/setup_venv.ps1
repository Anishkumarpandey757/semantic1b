# Text-to-SQL Virtual Environment Setup
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "TEXT-TO-SQL VIRTUAL ENVIRONMENT SETUP" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$VENV_NAME = "text2sql_env"
$VENV_PATH = ".\$VENV_NAME"

# Step 1: Check Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonCheck = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Found: $pythonCheck" -ForegroundColor Green
} else {
    Write-Host "  Python not found!" -ForegroundColor Red
    exit 1
}

# Step 2: Check existing venv
Write-Host ""
Write-Host "[2/6] Checking existing virtual environment..." -ForegroundColor Yellow
if (Test-Path $VENV_PATH) {
    Write-Host "  Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "  Delete and recreate? (y/N)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "  Deleting..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VENV_PATH
        Write-Host "  Deleted" -ForegroundColor Green
    } else {
        Write-Host "  Using existing environment" -ForegroundColor Cyan
        & "$VENV_PATH\Scripts\Activate.ps1"
        exit 0
    }
}

# Step 3: Create venv
Write-Host ""
Write-Host "[3/6] Creating virtual environment..." -ForegroundColor Yellow
python -m venv $VENV_NAME
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Created at: $VENV_PATH" -ForegroundColor Green
} else {
    Write-Host "  Failed to create venv" -ForegroundColor Red
    exit 1
}

# Step 4: Activate
Write-Host ""
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = "$VENV_PATH\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "  Activated" -ForegroundColor Green
} else {
    Write-Host "  Activation script not found" -ForegroundColor Red
    exit 1
}

# Step 5: Upgrade pip
Write-Host ""
Write-Host "[5/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "  Pip upgraded" -ForegroundColor Green

# Step 6: Install dependencies
Write-Host ""
Write-Host "[6/6] Installing dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
Write-Host ""

Write-Host "  Installing PyTorch (CPU)..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host "  Installing Transformers and PEFT..." -ForegroundColor Cyan
pip install transformers peft

Write-Host "  Installing Training tools..." -ForegroundColor Cyan
pip install datasets accelerate sentencepiece

Write-Host "  Installing Evaluation tools..." -ForegroundColor Cyan
pip install func-timeout

Write-Host ""
Write-Host "  All dependencies installed!" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Virtual environment: $VENV_PATH" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate in the future:" -ForegroundColor Yellow
Write-Host "  .\$VENV_NAME\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. python src\create_bird_semantic_dataset.py" -ForegroundColor White
Write-Host "  2. python src\phase3_train_lora_optimized.py" -ForegroundColor White
Write-Host "  3. python src\phase4_evaluate_execution_accuracy.py" -ForegroundColor White
Write-Host ""
