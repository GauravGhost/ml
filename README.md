# 🔬 Biometric Classification Project

A comprehensive machine learning project that trains and compares different CNN models for biometric classification including fingerprint, face, and iris recognition. **Works on macOS, Linux, and Windows!**

## 🎯 Features

- **3 Biometric Classifiers**: Fingerprint, Face, and Iris recognition
- **Multiple CNN Models**: ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception
- **Cross-Platform**: Native support for macOS, Linux, and Windows
- **Multiple Setup Options**: Bash, Batch files, PowerShell scripts
- **Automatic Detection**: Binary/Multi-class classification auto-detected from dataset
- **GPU Support**: Automatically uses GPU when available
- **Comprehensive Analysis**: Detailed performance analysis and visualizations for each classifier
- **Easy Deployment**: Simple model usage scripts for predictions
- **Modular Structure**: Organized into separate classifier modules
- **Unified Interface**: Single main.py script to run everything

## 📁 Project Structure

```
ml/
├── main.py                         # 🎯 Main project runner (unified interface)
├── classifiers/                    # 🔬 Classifier modules
│   ├── fingerprint/               # 👆 Fingerprint classification
│   │   ├── fingerprint_classifier.py
│   │   └── use_model.py
│   ├── face/                      # 👤 Face recognition
│   │   ├── face_classifier.py
│   │   └── use_face_model.py
│   └── iris/                      # 👁️ Iris recognition
│       ├── iris_classifier.py
│       └── use_iris_model.py
├── utils/                         # 🛠️ Shared utilities
│   └── analyze_results.py         # 📊 Results analysis & visualization
├── data/                          # 📂 Dataset storage
│   ├── fingerprint/
│   ├── face/
│   └── iris/
├── results/                       # 💾 All outputs saved here
│   ├── fingerprint/
│   ├── face/
│   └── iris/
├── setup.sh                       # 🛠️ Cross-platform setup
└── activate_env.sh               # 🔥 Cross-platform activation
```

## 🚀 Quick Start

### 🍎 macOS / 🐧 Linux

#### 1. Setup (Run Once)
```bash
./setup.sh                    # Sets up everything automatically
```

#### 2. Train and Analyze Models
```bash
source activate_env.sh                # Activate environment

# Run specific classifiers
python main.py -c fingerprint -a train    # Train fingerprint classifier
python main.py -c face -a train          # Train face classifier  
python main.py -c iris -a train          # Train iris classifier

# Use trained models
python main.py -c fingerprint -a use     # Use fingerprint model
python main.py -c face -a use           # Use face model
python main.py -c iris -a use           # Use iris model

# Analyze results
python main.py -a analyze               # Analyze all available results
python main.py -c fingerprint -a analyze # Analyze specific classifier
python analyze.py                       # Quick analysis (all classifiers)
python analyze.py -c fingerprint        # Quick analysis (specific)
```

### 🪟 Windows

**Recommended: Use Git Bash** (most reliable)

1. Install [Git for Windows](https://gitforwindows.org/)
2. Open Git Bash in project directory:
```bash
./setup.sh                    # Cross-platform setup
./activate_env.sh             # Cross-platform activation
python fingerprint_classifier.py
```

**Alternative: Command Prompt or PowerShell**
If you prefer native Windows commands, the bash scripts work in most cases, but you may need to:
- Disable Windows Store Python aliases (Settings > Apps > App execution aliases)
- Use `py` instead of `python` if needed

## 📋 Prerequisites

### Essential for All Platforms:
- **Python 3.8+** 
  - macOS: `brew install python` or download from [python.org](https://python.org)
  - Windows: Download from [python.org](https://python.org) (⚠️ Check "Add Python to PATH")
  - Linux: `sudo apt install python3 python3-pip` or similar

### Windows-Specific:
- **Git for Windows** (for Git Bash option): [git-scm.com](https://git-scm.com/)

### Optional (GPU acceleration):
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

### 3. Analyze Results (All Platforms)
```bash
# macOS/Linux/Git Bash
python analyze_results.py            

# Windows Command Prompt  
python analyze_results.py

# Windows PowerShell
python analyze_results.py
```

### 4. Use Trained Models (All Platforms)
```bash
python use_model.py                  # Demo for using models
```

## 🌟 Platform-Specific Tips

### 🍎 macOS Users
- Use `brew install python` for easy Python management
- Homebrew automatically handles PATH configuration
- GPU support via Metal Performance Shaders (built-in)

### 🐧 Linux Users  
- Most distributions work out of the box
- Use your package manager: `apt`, `yum`, `pacman`, etc.
- NVIDIA users: install CUDA toolkit for GPU acceleration

### 🪟 Windows Users
- **Git Bash recommended** for consistent experience
- **VS Code** provides excellent Python development experience
- **Windows Terminal** offers better PowerShell experience
- Enable GPU support with NVIDIA CUDA toolkit

## 📊 What You Get

### 🏆 Model Comparison
- Performance rankings of all 6 models
- Accuracy, Precision, Recall, F1-Score metrics
- Confusion matrices and ROC curves
- Training history visualizations

### 📈 Analysis Reports
- `model_comparison_results.csv` - Raw performance data
- `detailed_analysis_report.txt` - Human-readable analysis
- `model_performance_analysis.png` - Comprehensive charts
- `model_radar_chart.png` - Top 3 models comparison

### 🧠 Trained Models
- All 6 models saved as `.h5` files
- Ready for deployment and predictions
- Best model automatically identified

## 💡 Key Configuration

Edit `fingerprint_classifier.py` to customize:
```python
IMG_SIZE = (160, 160)    # Image size
BATCH_SIZE = 32          # Batch size  
EPOCHS = 10              # Training epochs
```

## 🎯 Typical Results

Based on training, you can expect:
- **InceptionV3**: ~99% accuracy (usually best)
- **Xception**: ~99% accuracy  
- **VGG16**: ~98% accuracy
- Other models: 60-98% depending on data

## 🔧 Troubleshooting

### General Issues
| Issue | Solution |
|-------|----------|
| No images found | Check dataset structure in `data/fingerprint/` |
| Out of memory | Reduce `BATCH_SIZE` in script |
| Slow training | Enable GPU or reduce `EPOCHS` |
| Poor accuracy | Check data quality & balance |

### Windows-Specific Issues
| Issue | Solution |
|-------|----------|
| "Python not found" | Reinstall Python, check "Add Python to PATH" |
| "Execution Policy Error" | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| "Permission denied" | Run Command Prompt/PowerShell as Administrator |
| Package install fails | Try: `pip install --user -r requirements.txt` |

### macOS/Linux Issues
| Issue | Solution |
|-------|----------|
| Permission denied | Run: `chmod +x setup.sh activate_env.sh` |
| Python not found | Install via package manager or python.org |
| pip not available | Run: `python3 -m ensurepip --upgrade` |

## 📋 Cross-Platform Requirements

- **Python**: 3.8+ (All platforms)
- **Memory**: At least 8GB RAM  
- **Storage**: 2GB disk space for models
- **TensorFlow**: 2.15+ (installed automatically)
- **GPU**: Optional, but recommended for faster training

### Platform-Specific:
- **Windows**: Visual C++ Redistributable (usually included)
- **macOS**: Xcode Command Line Tools: `xcode-select --install`  
- **Linux**: python3-dev package: `sudo apt install python3-dev`

## 🏗️ Architecture

Each model uses:
- Pre-trained weights (ImageNet)
- Frozen base layers
- Custom classification head:
  - GlobalAveragePooling2D
  - Dense(64, relu)
  - Dense(classes, softmax/sigmoid)

## 🚀 Production Usage

After training, use your best model:
```python
import tensorflow as tf
model = tf.keras.models.load_model('results/InceptionV3.h5')
# Your prediction code here
```

---