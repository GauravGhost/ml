# 🔬 Fingerprint Classification Project

## 🎯 Features

- **6 CNN Models**: ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception
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
├── setup.sh                    # 🛠️ One-time environment setup
├── activate_env.sh             # 🔥 Environment activation
├── data/fingerprint/           # 📂 Your dataset goes here
└── results/                    # 💾 All outputs saved here
```

## 🚀 Quick Start

### 1. Setup (Run Once)
```bash
./setup.sh                    # Sets up everything automatically
```

### 2. Train Models
```bash
source activate_env.sh                # Activate environment
python fingerprint_classifier.py     # Train all 6 models (simple!)
```

### 3. Analyze Results
```bash
python analyze_results.py            # Get comprehensive analysis
```

### 4. Use Trained Models
```bash
python use_model.py                  # Demo for using models
```

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

| Issue | Solution |
|-------|----------|
| No images found | Check dataset structure in `data/fingerprint/` |
| Out of memory | Reduce `BATCH_SIZE` in script |
| Slow training | Enable GPU or reduce `EPOCHS` |
| Poor accuracy | Check data quality & balance |

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.15+
- GPU (optional, but recommended)
- At least 8GB RAM
- 2GB disk space for models

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