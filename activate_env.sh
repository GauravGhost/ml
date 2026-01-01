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
# Check if we're on Windows (Git Bash/MINGW) or Linux/Mac
if [ -f "venv/Scripts/activate" ]; then
    # Windows path
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac path  
    source venv/bin/activate
else
    echo "❌ Activation script not found in venv/Scripts or venv/bin"
    exit 1
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
