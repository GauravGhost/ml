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
        if pip install -r requirements.txt; then
            print_success "All packages installed successfully"
        else
            print_warning "Some packages failed to install. Trying with latest versions..."
            print_status "Installing essential packages with latest versions..."
            pip install tensorflow numpy pandas scikit-learn matplotlib seaborn Pillow opencv-python tqdm jupyter ipykernel
            
            if [ $? -eq 0 ]; then
                print_success "Essential packages installed successfully"
            else
                print_error "Failed to install required packages. Please check your Python version and internet connection."
                exit 1
            fi
        fi
    else
        print_error "requirements.txt not found. Installing basic packages..."
        pip install tensorflow numpy pandas scikit-learn matplotlib Pillow
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

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for biometric classification environment

echo "🔥 Activating Biometric Classification Environment"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment based on platform
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo "🪟 Windows platform detected"
    source venv/Scripts/activate
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS platform detected"
    source venv/bin/activate
else
    echo "🐧 Linux platform detected"
    source venv/bin/activate
fi

echo "✅ Environment activated"

# Check GPU availability using our enhanced GPU detection
echo ""
echo "🔍 Checking GPU availability..."
python -c "
try:
    from utils.gpu_utils import setup_gpu, get_gpu_info
    setup_gpu()
    gpu_info = get_gpu_info()
    print(f'GPU Status: {gpu_info}')
except ImportError:
    print('GPU utilities will be available after first training run')
except Exception as e:
    print(f'GPU check: {str(e)}')
" 2>/dev/null || echo "GPU check will be performed during training"

echo ""
echo "📋 Available commands:"
echo ""
echo "📸 FACE ANTI-SPOOFING (Enhanced for 90%+ accuracy):"
echo "  python main.py -c face -a train            # Train face classification models"
echo "  python main.py -c face -a analyze          # Analyze face training results"
echo "  python main.py -c face -a use              # Use trained face models"
echo "  cd classifiers/face && python face_classifier.py   # Direct enhanced training"
echo ""
echo "👆 FINGERPRINT CLASSIFICATION:"
echo "  python main.py -c fingerprint -a train     # Train fingerprint classification models" 
echo "  python main.py -c fingerprint -a analyze   # Analyze fingerprint training results"
echo "  python main.py -c fingerprint -a use       # Use trained fingerprint models"
echo ""
echo "👁️  IRIS RECOGNITION:"
echo "  python main.py -c iris -a train            # Train iris classification models"
echo "  python main.py -c iris -a analyze          # Analyze iris training results"
echo "  python main.py -c iris -a use              # Use trained iris models"
echo ""
echo "📊 ANALYSIS TOOLS:"
echo "  python utils/analyze_results.py            # Compare all model results"
echo "  cat DATA_SETUP.md                         # View dataset setup guide"
echo ""
echo "🚀 ENHANCED FEATURES:"
echo "  • Advanced GPU detection (Mac Metal/CUDA/CPU fallback)"
echo "  • Face model optimized for 90%+ accuracy target"
echo "  • Two-phase training with transfer learning + fine-tuning"
echo "  • Dynamic class weighting for imbalanced datasets"
echo "  • Cross-platform compatibility (Mac/Windows/Linux)"
echo "  • Absolute path resolution - works from any directory"
echo "  • Enhanced data augmentation and regularization"
echo ""
echo "📁 Directory structure:"
echo "  data/face/         - Face dataset (live vs spoof attacks)"
echo "  data/fingerprint/  - Fingerprint dataset (genuine vs altered)"
echo "  data/iris/         - Iris dataset (real vs synthetic)"
echo "  results/           - Training results and model files"
echo "  utils/gpu_utils.py - Cross-platform GPU detection"
echo ""
echo "💡 TIPS:"
echo "  • Face model includes enhanced architecture for anti-spoofing"
echo "  • Check DATA_SETUP.md for detailed dataset organization"
echo "  • GPU acceleration automatically configured if available"
echo "  • Compatible with TensorFlow 2.15.0 + TensorFlow-Metal"
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
# Dataset Setup Instructions for Biometric Classification

## Required Directory Structure

### Face Classification Dataset
```
data/face/
├── live_subject_images/     # Real face images
│   ├── person1_image1.jpg
│   ├── person2_image1.jpg
│   └── ...
├── bobblehead_images/       # Bobblehead attack images
│   ├── DSLR/
│   ├── IPHONE14/
│   └── SAMSUNG_S9/
├── Full_cloth_mask_images/  # Cloth mask attack images
│   └── ...
└── [other_attack_types]/    # Various spoof attacks
```

### Fingerprint Classification Dataset
```
data/fingerprint/
├── Real/               # Genuine fingerprints
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── Altered/            # Altered fingerprints
│   ├── image1.jpg
│   └── ...
└── SOCOFing/          # SOCOFING dataset
    └── ...
```

### Iris Classification Dataset
```
data/iris/
├── genuine/            # Real iris images
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── spoofed/            # Fake iris images
    ├── image1.jpg
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
3. **Consistent Naming**: Use descriptive folder names
4. **Minimum Size**: At least 100 images per class recommended for good performance
5. **Image Size**: Images will be automatically resized to 224x224 pixels

## Training Commands

### Face Anti-Spoofing (Live vs Spoof Detection)
```bash
# Train all 6 models with enhanced architecture for 90%+ accuracy
python main.py -c face -a train

# Analyze results and get detailed performance report
python main.py -c face -a analyze

# Use the best trained model for predictions
python main.py -c face -a use
```

### Fingerprint Classification (Genuine vs Fake)
```bash
# Train all models
python main.py -c fingerprint -a train

# Analyze results
python main.py -c fingerprint -a analyze

# Use trained models
python main.py -c fingerprint -a use
```

### Iris Recognition (Real vs Synthetic)
```bash
# Train all models
python main.py -c iris -a train

# Analyze results
python main.py -c iris -a analyze

# Use trained models
python main.py -c iris -a use
```

## GPU Acceleration

The system automatically detects and configures GPU acceleration:
- **Mac (Apple Silicon)**: Uses Metal Performance Shaders
- **Windows/Linux**: Uses NVIDIA CUDA (if available)
- **Fallback**: Uses CPU if GPU not available

## Performance Optimization

For **90%+ accuracy** (especially face classification):
- Enhanced data augmentation (rotation, brightness, zoom)
- Two-phase training (transfer learning + fine-tuning)
- Dynamic class weighting for imbalanced datasets
- Advanced callbacks (early stopping, learning rate scheduling)
- Optimized architecture with batch normalization and dropout

## Troubleshooting

- **"No images found"**: Check that images are in subdirectories, not directly in the main folder
- **"Out of memory"**: Reduce BATCH_SIZE in the classifier files
- **"Slow training"**: Ensure GPU drivers are installed (NVIDIA for Windows/Linux, TensorFlow-Metal for Mac)
- **"Path not found"**: The system now uses absolute paths, should work from any directory
- **"TensorFlow errors"**: Compatible with TensorFlow 2.15.0, check requirements.txt
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