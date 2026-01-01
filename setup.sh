#!/bin/bash

# Fingerprint Classification Environment Setup Script
# This script sets up the environment for local training

set -e  # Exit on any error

echo "🚀 Setting up Fingerprint Classification Environment"
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
        # Windows-specific: try to find actual Python executable
        if command -v py &> /dev/null; then
            # Try Windows Python Launcher first
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
    $PYTHON_CMDtatus "Setting up virtual environment..."
    
    VENV_DIR="venv"
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf "$VENV_DIR"
    fi
    
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    print_status "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        source "$VENV_DIR/Scripts/activate"
    else
        source "$VENV_DIR/bin/activate"
    fi
    
    print_status "Upgrading pip in virtual environment..."
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Install requirements
install_requirements() {
    print_status "Installing Python packages..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
        print_success "All packages installed successfully"
    else
        print_error "requirements.txt not found. Installing basic packages..."
        pip install tensorflow numpy pandas scikit-learn matplotlib Pillow
    fi
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create data directory
    mkdir -p data/fingerprint
    mkdir -p results
    mkdir -p logs
    
    print_success "Directory structure created:"
    echo "  📁 data/fingerprint/  - Place your dataset here"
    echo "  📁 results/          - Training results will be saved here"
    echo "  📁 logs/             - Log files will be stored here"
}

# Check GPU availability (optional)
check_gpu() {
    print_status "Checking GPU availability..."
    
    if $PYTHON_CMD -c "import tensorflow as tf; print('GPU available:', len(tf.config.experimental.list_physical_devices('GPU')) > 0)" 2>/dev/null; then
        print_success "GPU check completed"
    else
        print_warning "Could not check GPU availability. TensorFlow may not be installed yet."
    fi
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for fingerprint classification environment

echo "🔥 Activating Fingerprint Classification Environment"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Environment activated"
echo ""
echo "📋 Available commands:"
echo "  python fingerprint_classifier.py           # Train all models"
echo "  python analyze_results.py                 # Analyze training results"
echo "  python use_model.py                       # Use trained models"
echo ""
echo "📁 Directory structure:"
echo "  data/fingerprint/  - Place your dataset here"
echo "  results/           - Results will be saved here"
echo ""
echo "To deactivate the environment, run: deactivate"
EOF

    chmod +x activate_env.sh
    print_success "Activation script created: activate_env.sh"
}

# Create sample dataset structure info
create_dataset_info() {
    print_status "Creating dataset structure information..."
    
    cat > DATA_SETUP.md << 'EOF'
# Dataset Setup Instructions

## Required Directory Structure

Your dataset should be organized as follows:

```
data/fingerprint/
├── class1/          (e.g., "genuine" or "real")
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── class2/          (e.g., "fake" or "spoofed")
    ├── image3.jpg
    ├── image4.png
    └── ...
```

## Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Dataset Guidelines

1. **Balanced Classes**: Try to have roughly equal numbers of images in each class
2. **Image Quality**: Use clear, well-lit images for best results
3. **Consistent Naming**: Use descriptive folder names (e.g., "real_fingerprints", "fake_fingerprints")
4. **Minimum Size**: At least 100 images per class recommended
5. **Image Size**: Images will be automatically resized, but try to use consistent aspect ratios

## Example Commands

```bash
# Train all 6 models (ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception)
python fingerprint_classifier.py

# After training, analyze results and get performance report
python analyze_results.py

# Use the best trained model for predictions
python use_model.py
```

## Troubleshooting

- **"No images found"**: Check that images are in subdirectories, not directly in the main folder
- **"Out of memory"**: Reduce BATCH_SIZE or IMG_SIZE in fingerprint_classifier.py
- **"Slow training"**: Enable GPU or reduce EPOCHS in fingerprint_classifier.py
EOF

    print_success "Dataset setup guide created: DATA_SETUP.md"
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
    create_activation_script
    create_dataset_info
    
    echo
    print_success "🎉 Setup completed successfully!"
    echo
    echo "🔥 Next Steps:"
    echo "1. Place your dataset in data/fingerprint/ following the structure in DATA_SETUP.md"
    echo "2. Activate the environment: source activate_env.sh"
    echo "3. Run training: python fingerprint_classifier.py"
    echo
    echo "💡 Useful commands:"
    echo "  ./activate_env.sh                          # Activate environment"
    echo "  python fingerprint_classifier.py           # Train all models"
    echo "  python analyze_results.py                 # Analyze training results"
    echo "  python use_model.py                       # Use trained models"
    echo
    echo "📚 Read DATA_SETUP.md for dataset setup instructions"
}

# Check if script is being run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi