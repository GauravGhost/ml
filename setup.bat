@echo off
:: Fingerprint Classification Environment Setup Script - Windows Batch Version
:: This script sets up the environment for local training on Windows

echo.
echo 🚀 Setting up Fingerprint Classification Environment (Windows)
echo ===========================================================

echo [INFO] Starting environment setup...

:: Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Python found and working
    
    :: Check if version is compatible
    python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo [SUCCESS] Python version is compatible (3.8+)
    ) else (
        echo [ERROR] Python 3.8 or higher is required
        echo Please download Python from https://python.org
        pause
        exit /b 1
    )
) else (
    echo [ERROR] Python is not installed or not in PATH
    echo Please download and install Python 3.8+ from https://python.org
    echo Make sure to check 'Add Python to PATH' during installation
    pause
    exit /b 1
)

:: Check if pip is installed
echo [INFO] Checking pip installation...
pip --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] pip found and working
) else (
    echo [WARNING] pip not available. Installing pip...
    python -m ensurepip --upgrade
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install pip
        pause
        exit /b 1
    )
)

:: Setup virtual environment
echo [INFO] Setting up virtual environment...
set VENV_DIR=venv

if exist "%VENV_DIR%" (
    echo [WARNING] Virtual environment already exists. Removing old one...
    rmdir /s /q "%VENV_DIR%"
)

echo [INFO] Creating virtual environment...
python -m venv "%VENV_DIR%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [INFO] Upgrading pip in virtual environment...
pip install --upgrade pip

echo [SUCCESS] Virtual environment created and activated

:: Install requirements
echo [INFO] Installing Python packages...
if exist "requirements.txt" (
    echo [INFO] Installing from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% equ 0 (
        echo [SUCCESS] All packages installed successfully
    ) else (
        echo [ERROR] Some packages failed to install
        pause
        exit /b 1
    )
) else (
    echo [WARNING] requirements.txt not found. Installing basic packages...
    pip install tensorflow numpy pandas scikit-learn matplotlib Pillow
)

:: Create directories
echo [INFO] Creating directory structure...
if not exist "data\fingerprint" mkdir "data\fingerprint"
if not exist "results" mkdir "results"
if not exist "logs" mkdir "logs"

echo [SUCCESS] Directory structure created:
echo   📁 data\fingerprint\  - Place your dataset here
echo   📁 results\          - Training results will be saved here
echo   📁 logs\             - Log files will be stored here

:: Create activation script
echo [INFO] Creating activation script...
(
echo @echo off
echo :: Activation script for fingerprint classification environment
echo.
echo echo 🔥 Activating Fingerprint Classification Environment
echo echo ==================================================
echo.
echo :: Check if virtual environment exists
echo if not exist "venv" ^(
echo     echo ❌ Virtual environment not found. Run setup.bat first.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo :: Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo echo ✅ Environment activated
echo echo.
echo echo 📋 Available commands:
echo echo   python fingerprint_classifier.py           # Train all models
echo echo   python analyze_results.py                 # Analyze training results
echo echo   python use_model.py                       # Use trained models
echo echo.
echo echo 📁 Directory structure:
echo echo   data\fingerprint\  - Place your dataset here
echo echo   results\           - Results will be saved here
echo echo.
echo echo To deactivate the environment, run: deactivate
echo echo.
echo cmd /k
) > activate_env.bat

echo [SUCCESS] Activation script created: activate_env.bat

echo.
echo [SUCCESS] 🎉 Setup completed successfully!
echo.
echo 🔥 Next Steps:
echo 1. Place your dataset in data\fingerprint\ following the structure in DATA_SETUP.md
echo 2. Activate the environment: activate_env.bat
echo 3. Run training: python fingerprint_classifier.py
echo.
echo 💡 Useful commands:
echo   activate_env.bat                          # Activate environment
echo   python fingerprint_classifier.py         # Train all models
echo   python analyze_results.py               # Analyze training results
echo   python use_model.py                     # Use trained models
echo.
echo 📚 Read DATA_SETUP.md for dataset setup instructions

pause