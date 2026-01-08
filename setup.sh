#!/bin/bash

# Biometric Classification Environment Setup Script
# This script sets up the environment for Face, Fingerprint, and Iris classification training

set -e  # Exit on any error

echo "🚀 Setting up Biometric Classification Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
check_python() {
    print_status "Checking Python installation..."
    
    # Check for python3 first (Linux/macOS), then python (Windows)
    PYTHON_CMD=""
    
    # For Windows (Git Bash), try different approaches to find Python
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        # Windows-specific: try to find Python 3.12 first (TensorFlow compatible)
        if command -v py &> /dev/null && py -3.12 --version &> /dev/null; then
            # Use Python 3.12 specifically for TensorFlow compatibility
            PYTHON_CMD="py -3.12"
        elif command -v py &> /dev/null; then
            # Try Windows Python Launcher with default version
            PYTHON_CMD="py"
        elif command -v python.exe &> /dev/null; then
            # Try python.exe directly
            PYTHON_CMD="python.exe"
        elif which python 2>/dev/null | grep -v "not found" | grep -v "Microsoft" &> /dev/null; then
            # Try python but avoid Microsoft Store alias
            PYTHON_CMD="python"
        else
            print_error "Python is not properly installed or configured."
            echo "Windows issue detected: Microsoft Store alias is interfering."
            echo ""
            echo "Please fix this by either:"
            echo "1. Disable Python app execution aliases:"
            echo "   Settings > Apps > Advanced app settings > App execution aliases"
            echo "   Turn OFF 'python.exe' and 'python3.exe'"
            echo ""
            echo "2. Or reinstall Python from python.org and ensure 'Add to PATH' is checked"
            exit 1
        fi
    else
        # Linux/macOS: try python3 first, then python
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            print_error "Python is not installed or not in PATH."
            print_status "On macOS, you can install Python using:"
            echo "  brew install python"
            echo "  or download from https://python.org"
            exit 1
        fi
    fi
    
    # Test if the Python command actually works
    if ! $PYTHON_CMD --version &> /dev/null; then
        print_error "Python command '$PYTHON_CMD' found but not working properly."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found (using: $PYTHON_CMD)"
    
    # Check if version is >= 3.8
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_success "Python version is compatible (>=3.8)"
    else
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    
    # Use same python command as discovered above
    PIP_CMD=""
    if command -v pip3 &> /dev/null && [ "$PYTHON_CMD" = "python3" ]; then
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
    else
        print_error "pip is not installed. Installing pip..."
        $PYTHON_CMD -m ensurepip --upgrade
        PIP_CMD="pip"
    fi
    
    if command -v $PIP_CMD &> /dev/null; then
        PIP_VERSION=$($PIP_CMD --version | cut -d' ' -f2)
        print_success "pip $PIP_VERSION found"
    fi
}

# Create virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    VENV_DIR="venv"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf "$VENV_DIR"
    fi
    
    print_status "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    print_status "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        source "$VENV_DIR/Scripts/activate"
        VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
    else
        source "$VENV_DIR/bin/activate"
        VENV_PYTHON="$VENV_DIR/bin/python"
    fi
    
    print_status "Upgrading pip in virtual environment..."
    $VENV_PYTHON -m pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Install requirements
install_requirements() {
    print_status "Installing Python packages..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        if pip install --only-binary=all -r requirements.txt; then
            print_success "All packages installed successfully"
        else
            print_warning "Some packages failed to install. Trying with latest versions..."
            print_status "Installing essential packages with latest versions..."
            pip install --only-binary=all tensorflow numpy pandas scikit-learn matplotlib seaborn Pillow opencv-python tqdm jupyter ipykernel
            
            if [ $? -eq 0 ]; then
                print_success "Essential packages installed successfully"
            else
                print_error "Failed to install required packages. Please check your Python version and internet connection."
                exit 1
            fi
        fi
    else
        print_error "requirements.txt not found. Installing basic packages..."
        pip install --only-binary=all tensorflow numpy pandas scikit-learn matplotlib Pillow
    fi
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create data directories for all biometric types
    mkdir -p data/face
    mkdir -p data/fingerprint
    mkdir -p data/iris
    mkdir -p results/face
    mkdir -p results/fingerprint
    mkdir -p results/iris
    mkdir -p utils
    
    print_success "Directory structure created:"
    echo "  📁 data/face/        - Place your face dataset here"
    echo "  📁 data/fingerprint/ - Place your fingerprint dataset here"
    echo "  📁 data/iris/        - Place your iris dataset here"
    echo "  📁 results/          - Training results will be saved here"
    echo "  📁 utils/            - Utility scripts and GPU detection"
}

# Check GPU availability with enhanced cross-platform detection
check_gpu() {
    print_status "Checking GPU availability..."
    
    # Test our enhanced GPU detection
    if $PYTHON_CMD -c "
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'utils'))
try:
    from gpu_utils import setup_gpu_acceleration, get_device_info
    gpu_available = setup_gpu_acceleration()
    devices = get_device_info()
    if gpu_available:
        print('✅ GPU acceleration is available and configured')
    else:
        print('⚠️ GPU not available, will use CPU (slower but works)')
except Exception as e:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'✅ Found {len(gpus)} GPU device(s)')
    else:
        print('⚠️ No GPU detected, will use CPU')
" 2>/dev/null; then
        print_success "GPU detection completed"
    else
        print_warning "Could not check GPU availability. Will be checked during first training run."
    fi
}

# Main setup function
main() {
    echo
    print_status "Starting environment setup..."
    echo
    
    # Run setup steps
    check_python
    check_pip
    setup_venv
    install_requirements
    create_directories
    check_gpu
    
    echo
    print_success "🎉 Setup completed successfully!"
}

# Check if script is being run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi