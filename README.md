# 🔬 Fingerprint Classification Project

A comprehensive machine learning project that trains and compares 6 different CNN models for fingerprint classification. **Works on macOS, Linux, and Windows!**

## 🎯 Features

- **6 CNN Models**: ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception
- **Cross-Platform**: Native support for macOS, Linux, and Windows
- **Multiple Setup Options**: Bash, Batch files, PowerShell scripts
- **Automatic Detection**: Binary/Multi-class classification auto-detected from dataset
- **GPU Support**: Automatically uses GPU when available
- **Comprehensive Analysis**: Detailed performance analysis and visualizations
- **Easy Deployment**: Simple model usage scripts for predictions

## 📁 Project Structure

```
ml/
├── fingerprint_classifier.py    # 🎯 Main training script
├── analyze_results.py           # 📊 Results analysis & visualization
├── use_model.py                # 🧠 Model usage demo
├── setup.sh                    # 🛠️ Cross-platform bash setup
├── setup.bat                   # 🪟 Windows Command Prompt setup
├── setup.ps1                   # ⚡ Windows PowerShell setup
├── activate_env.sh             # 🔥 Cross-platform activation
├── activate_env.bat            # 🪟 Windows batch activation
├── activate_env.ps1            # ⚡ PowerShell activation
├── data/fingerprint/           # 📂 Your dataset goes here
└── results/                    # 💾 All outputs saved here
```

## 🚀 Quick Start

### 🍎 macOS / 🐧 Linux

#### 1. Setup (Run Once)
```bash
./setup.sh                    # Sets up everything automatically
```

#### 2. Train Models
```bash
source activate_env.sh                # Activate environment
python fingerprint_classifier.py     # Train all 6 models
```

### 🪟 Windows

Choose your preferred method:

#### Option 1: Git Bash (Recommended)
1. Install [Git for Windows](https://gitforwindows.org/)
2. Open Git Bash in project directory:
```bash
./setup.sh                    # Cross-platform setup
./activate_env.sh             # Cross-platform activation
python fingerprint_classifier.py
```

#### Option 2: Command Prompt
```cmd
setup.bat                     # Windows batch setup
activate_env.bat              # Windows batch activation  
python fingerprint_classifier.py
```

#### Option 3: PowerShell
```powershell
# Enable script execution (first time only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\setup.ps1                   # PowerShell setup
.\activate_env.ps1            # PowerShell activation
python fingerprint_classifier.py
```

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