@echo off
:: Activation script for fingerprint classification environment

echo 🔥 Activating Fingerprint Classification Environment
echo ==================================================

:: Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

echo ✅ Environment activated
echo.
echo 📋 Available commands:
echo   python fingerprint_classifier.py           # Train all models
echo   python analyze_results.py                 # Analyze training results
echo   python use_model.py                       # Use trained models
echo.
echo 📁 Directory structure:
echo   data\fingerprint\  - Place your dataset here
echo   results\           - Results will be saved here
echo.
echo To deactivate the environment, run: deactivate
echo.
cmd /k