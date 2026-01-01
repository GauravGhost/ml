# Fingerprint Classification Environment Setup Script - PowerShell Version
# This script sets up the environment for local training on Windows

param(
    [switch]$Force
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚀 Setting up Fingerprint Classification Environment (PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check Python installation
function Test-Python {
    Write-Status "Checking Python installation..."
    
    try {
        $pythonVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python $($pythonVersion.Split(' ')[1]) found"
            
            # Check if version is >= 3.8
            $versionCheck = & python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Python version is compatible (>=3.8)"
            } else {
                Write-Error "Python 3.8 or higher is required. Found: $($pythonVersion.Split(' ')[1])"
                Write-Host "Please download Python from https://python.org" -ForegroundColor Yellow
                throw "Incompatible Python version"
            }
        } else {
            throw "Python not found"
        }
    }
    catch {
        Write-Error "Python is not installed or not in PATH."
        Write-Host "Please download and install Python 3.8+ from https://python.org" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        throw "Python not available"
    }
}

# Function to check pip installation
function Test-Pip {
    Write-Status "Checking pip installation..."
    
    try {
        $pipVersion = & pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "pip found"
        } else {
            throw "pip not found"
        }
    }
    catch {
        Write-Warning "pip is not available. Installing pip..."
        & python -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install pip"
        }
    }
}

# Function to setup virtual environment
function New-VirtualEnvironment {
    Write-Status "Setting up virtual environment..."
    
    $venvDir = "venv"
    
    if (Test-Path $venvDir) {
        if ($Force) {
            Write-Warning "Virtual environment already exists. Removing old one..."
            Remove-Item -Path $venvDir -Recurse -Force
        } else {
            Write-Warning "Virtual environment already exists. Use -Force to recreate."
            return
        }
    }
    
    Write-Status "Creating virtual environment..."
    & python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    
    Write-Status "Activating virtual environment..."
    & "$venvDir\Scripts\Activate.ps1"
    
    Write-Status "Upgrading pip in virtual environment..."
    & pip install --upgrade pip
    
    Write-Success "Virtual environment created and activated"
}

# Function to install requirements
function Install-Requirements {
    Write-Status "Installing Python packages..."
    
    if (Test-Path "requirements.txt") {
        Write-Status "Installing from requirements.txt..."
        & pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All packages installed successfully"
        } else {
            throw "Some packages failed to install"
        }
    } else {
        Write-Warning "requirements.txt not found. Installing basic packages..."
        & pip install tensorflow numpy pandas scikit-learn matplotlib Pillow
    }
}

# Function to create directories
function New-ProjectDirectories {
    Write-Status "Creating directory structure..."
    
    $directories = @("data\fingerprint", "results", "logs")
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directory structure created:"
    Write-Host "  📁 data\fingerprint\  - Place your dataset here" -ForegroundColor White
    Write-Host "  📁 results\          - Training results will be saved here" -ForegroundColor White
    Write-Host "  📁 logs\             - Log files will be stored here" -ForegroundColor White
}

# Function to create activation script
function New-ActivationScript {
    Write-Status "Creating PowerShell activation script..."
    
    $activationScript = @"
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
"@

    $activationScript | Out-File -FilePath "activate_env.ps1" -Encoding UTF8
    Write-Success "PowerShell activation script created: activate_env.ps1"
}

# Main execution
try {
    Write-Host ""
    Write-Status "Starting environment setup..."
    Write-Host ""
    
    # Run setup steps
    Test-Python
    Test-Pip
    New-VirtualEnvironment
    Install-Requirements
    New-ProjectDirectories
    New-ActivationScript
    
    Write-Host ""
    Write-Success "🎉 Setup completed successfully!"
    Write-Host ""
    Write-Host "🔥 Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Place your dataset in data\fingerprint\ following the structure in DATA_SETUP.md" -ForegroundColor White
    Write-Host "2. Activate the environment: .\activate_env.ps1" -ForegroundColor White
    Write-Host "3. Run training: python fingerprint_classifier.py" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 Useful commands:" -ForegroundColor Cyan
    Write-Host "  .\activate_env.ps1                        # Activate environment" -ForegroundColor White
    Write-Host "  python fingerprint_classifier.py          # Train all models" -ForegroundColor White
    Write-Host "  python analyze_results.py                # Analyze training results" -ForegroundColor White
    Write-Host "  python use_model.py                      # Use trained models" -ForegroundColor White
    Write-Host ""
    Write-Host "📚 Read DATA_SETUP.md for dataset setup instructions" -ForegroundColor Yellow
    
} catch {
    Write-Error "Setup failed: $($_.Exception.Message)"
    Write-Host "Please check the error messages above and try again." -ForegroundColor Red
    exit 1
}

Read-Host "Press Enter to continue"