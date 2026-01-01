# Activation script for fingerprint classification environment

Write-Host "🔥 Activating Fingerprint Classification Environment" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
& venv\Scripts\Activate.ps1

Write-Host "✅ Environment activated" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Available commands:" -ForegroundColor Yellow
Write-Host "  python fingerprint_classifier.py           # Train all models" -ForegroundColor White
Write-Host "  python analyze_results.py                 # Analyze training results" -ForegroundColor White
Write-Host "  python use_model.py                       # Use trained models" -ForegroundColor White
Write-Host ""
Write-Host "📁 Directory structure:" -ForegroundColor Yellow
Write-Host "  data\fingerprint\  - Place your dataset here" -ForegroundColor White
Write-Host "  results\           - Results will be saved here" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate the environment, run: deactivate" -ForegroundColor Cyan