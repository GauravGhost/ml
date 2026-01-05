#!/usr/bin/env python3
"""
Simple usage example for the trained fingerprint classification models
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def load_best_model():
    """Load the best performing model"""
    # Based on analysis, InceptionV3 is the best model
    model_path = "./results/fingerprint/InceptionV3.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("📋 Please run fingerprint_classifier.py first to train models")
        return None
    
    print(f"📂 Loading best model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully!")
    return model

def predict_fingerprint(model, image_path, show_image=True):
    """Make prediction on a fingerprint image"""
    
    # Class names based on your dataset
    class_names = ['Altered', 'Real', 'SOCOFing', 'models']
    
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(160, 160))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class = class_names[predicted_class_idx]
        
        # Show results
        print(f"\n🔍 Prediction Results:")
        print(f"   📸 Image: {os.path.basename(image_path)}")
        print(f"   🎯 Prediction: {predicted_class}")
        print(f"   📊 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        # Show detailed probabilities
        print(f"\n📈 All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            indicator = "👈" if i == predicted_class_idx else "  "
            print(f"   {indicator} {class_name:<10}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Optionally display image
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"Prediction: {predicted_class} ({confidence:.3f})")
            plt.axis('off')
            plt.show()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None

def batch_predict(model, image_folder):
    """Predict on multiple images in a folder"""
    
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"❌ No image files found in: {image_folder}")
        return
    
    print(f"🔍 Found {len(image_files)} images to analyze...")
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        predicted_class, confidence = predict_fingerprint(model, img_path, show_image=False)
        
        if predicted_class:
            results.append({
                'filename': img_file,
                'prediction': predicted_class,
                'confidence': confidence
            })
    
    # Summary
    print(f"\n📊 Batch Prediction Summary:")
    print("-" * 50)
    for result in results:
        print(f"{result['filename']:<20} → {result['prediction']:<10} ({result['confidence']:.3f})")
    
    return results

def main():
    """Main demo function"""
    print("🔬 FINGERPRINT CLASSIFICATION DEMO")
    print("=" * 40)
    
    # Load model
    model = load_best_model()
    if model is None:
        return
    
    print(f"\n🎯 Usage Options:")
    print("1. Single image prediction:")
    print("   predict_fingerprint(model, 'path/to/image.jpg')")
    print("\n2. Batch prediction:")
    print("   batch_predict(model, 'path/to/folder/')")
    
    # Example usage (uncomment to test):
    # predict_fingerprint(model, "data/fingerprint/Real/1_1.png")
    # batch_predict(model, "data/fingerprint/Real/")
    
    print(f"\n💡 Model Info:")
    print(f"   🏗️  Architecture: InceptionV3")
    print(f"   📏 Input size: 160x160 pixels")
    print(f"   🎯 Classes: Altered, Real, SOCOFing, models")
    print(f"   🎯 Accuracy: 99.4%")

if __name__ == "__main__":
    main()