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
echo "📁 Directory structure:"
echo "  data/face/         - Face dataset (live vs spoof attacks)"
echo "  data/fingerprint/  - Fingerprint dataset (genuine vs altered)"
echo "  data/iris/         - Iris dataset (real vs synthetic)"
echo "  results/           - Training results and model files"
echo "  utils/gpu_utils.py - Cross-platform GPU detection"
echo ""
echo "To deactivate the environment, run: deactivate"
