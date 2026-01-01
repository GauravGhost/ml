#!/bin/bash
# Activation script for fingerprint classification environment

echo "🔥 Activating Fingerprint Classification Environment"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate virtual environment (cross-platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

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
