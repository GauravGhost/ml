@echo off
:: Fingerprint Classification Environment Setup Script - Windows Batch Version
:: This script sets up the environment for local training on Windows

setlocal enabledelayedexpansion

echo.
echo 🚀 Setting up Fingerprint Classification Environment (Windows)
echo ===========================================================

:: Function to print status messages
call :print_status "Starting environment setup..."

:: Check if Python is installed
call :check_python
if !ERRORLEVEL! neq 0 exit /b 1

:: Check if pip is installed
call :check_pip
if !ERRORLEVEL! neq 0 exit /b 1

:: Setup virtual environment
call :setup_venv
if !ERRORLEVEL! neq 0 exit /b 1

:: Install requirements
call :install_requirements
if !ERRORLEVEL! neq 0 exit /b 1

:: Create directories
call :create_directories

:: Create activation script
call :create_activation_script

:: Create dataset info
call :create_dataset_info

echo.
call :print_success "🎉 Setup completed successfully!"
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
goto :eof

:: Function to check Python installation
:check_python
call :print_status "Checking Python installation..."
python --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    call :print_success "Python !PYTHON_VERSION! found"
    
    :: Check if version is >= 3.8
    python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        call :print_success "Python version is compatible (>=3.8)"
    ) else (
        call :print_error "Python 3.8 or higher is required. Found: !PYTHON_VERSION!"
        echo Please download Python from https://python.org
        exit /b 1
    )
) else (
    call :print_error "Python is not installed or not in PATH."
    echo Please download and install Python 3.8+ from https://python.org
    echo Make sure to check 'Add Python to PATH' during installation
    exit /b 1
)
goto :eof

:: Function to check pip installation
:check_pip
call :print_status "Checking pip installation..."
pip --version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    for /f "tokens=2" %%v in ('pip --version 2^>^&1') do set PIP_VERSION=%%v
    call :print_success "pip !PIP_VERSION! found"
) else (
    call :print_error "pip is not installed. Installing pip..."
    python -m ensurepip --upgrade
    if !ERRORLEVEL! neq 0 (
        call :print_error "Failed to install pip"
        exit /b 1
    )
)
goto :eof

:: Function to setup virtual environment
:setup_venv
call :print_status "Setting up virtual environment..."

set VENV_DIR=venv

if exist "!VENV_DIR!" (
    call :print_warning "Virtual environment already exists. Removing old one..."
    rmdir /s /q "!VENV_DIR!"
)

call :print_status "Creating virtual environment..."
python -m venv "!VENV_DIR!"
if !ERRORLEVEL! neq 0 (
    call :print_error "Failed to create virtual environment"
    exit /b 1
)

call :print_status "Activating virtual environment..."
call "!VENV_DIR!\Scripts\activate.bat"

call :print_status "Upgrading pip in virtual environment..."
pip install --upgrade pip

call :print_success "Virtual environment created and activated"
goto :eof

:: Function to install requirements
:install_requirements
call :print_status "Installing Python packages..."

if exist "requirements.txt" (
    call :print_status "Installing from requirements.txt..."
    pip install -r requirements.txt
    if !ERRORLEVEL! equ 0 (
        call :print_success "All packages installed successfully"
    ) else (
        call :print_error "Some packages failed to install"
        exit /b 1
    )
) else (
    call :print_error "requirements.txt not found. Installing basic packages..."
    pip install tensorflow numpy pandas scikit-learn matplotlib Pillow
)
goto :eof

:: Function to create directories
:create_directories
call :print_status "Creating directory structure..."

if not exist "data\fingerprint" mkdir "data\fingerprint"
if not exist "results" mkdir "results"
if not exist "logs" mkdir "logs"

call :print_success "Directory structure created:"
echo   📁 data\fingerprint\  - Place your dataset here
echo   📁 results\          - Training results will be saved here
echo   📁 logs\             - Log files will be stored here
goto :eof

:: Function to create activation script
:create_activation_script
call :print_status "Creating activation script..."

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

call :print_success "Activation script created: activate_env.bat"
goto :eof

:: Function to create dataset info
:create_dataset_info
call :print_status "Creating dataset structure information..."

:: Create the same DATA_SETUP.md but with Windows paths
(
echo # Dataset Setup Instructions
echo.
echo ## Required Directory Structure
echo.
echo Your dataset should be organized as follows:
echo.
echo ```
echo data\fingerprint\
echo ├── class1\          ^(e.g., "genuine" or "real"^)
echo │   ├── image1.jpg
echo │   ├── image2.png
echo │   └── ...
echo └── class2\          ^(e.g., "fake" or "spoofed"^)
echo     ├── image3.jpg
echo     ├── image4.png
echo     └── ...
echo ```
echo.
echo ## Supported Image Formats
echo - JPEG ^(.jpg, .jpeg^)
echo - PNG ^(.png^)
echo - BMP ^(.bmp^)
echo - TIFF ^(.tiff, .tif^)
echo.
echo ## Dataset Guidelines
echo.
echo 1. **Balanced Classes**: Try to have roughly equal numbers of images in each class
echo 2. **Image Quality**: Use clear, well-lit images for best results
echo 3. **Consistent Naming**: Use descriptive folder names ^(e.g., "real_fingerprints", "fake_fingerprints"^)
echo 4. **Minimum Size**: At least 100 images per class recommended
echo 5. **Image Size**: Images will be automatically resized, but try to use consistent aspect ratios
echo.
echo ## Example Commands ^(Windows^)
echo.
echo ```batch
echo REM Train all 6 models ^(ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception^)
echo python fingerprint_classifier.py
echo.
echo REM After training, analyze results and get performance report
echo python analyze_results.py
echo.
echo REM Use the best trained model for predictions
echo python use_model.py
echo ```
echo.
echo ## Troubleshooting
echo.
echo - **"No images found"**: Check that images are in subdirectories, not directly in the main folder
echo - **"Out of memory"**: Reduce BATCH_SIZE or IMG_SIZE in fingerprint_classifier.py
echo - **"Slow training"**: Enable GPU or reduce EPOCHS in fingerprint_classifier.py
echo - **"Python not found"**: Make sure Python is installed and added to PATH
echo - **"Permission denied"**: Run Command Prompt as Administrator
) > DATA_SETUP.md

call :print_success "Dataset setup guide created: DATA_SETUP.md"
goto :eof

:: Utility functions for colored output
:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof