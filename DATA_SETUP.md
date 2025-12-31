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
