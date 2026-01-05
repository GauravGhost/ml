#!/usr/bin/env python3
"""
Iris Classifier Usage Script - For making predictions on new iris images
"""

import tensorflow as tf
import numpy as np
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

class IrisPredictor:
    def __init__(self, model_path="./results/iris"):
        self.model_path = model_path
        self.model = None
        self.class_indices = None
        self.img_size = (224, 224)
        
    def load_best_model(self):
        """Load the best performing model"""
        # Find all .h5 model files
        model_files = glob.glob(os.path.join(self.model_path, "*.h5"))
        
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {self.model_path}")
        
        # For now, we'll use the first available model
        model_file = model_files[0]
        print(f"📦 Loading model: {model_file}")
        
        self.model = load_model(model_file)
        print(f"✅ Model loaded successfully!")
        
        model_name = os.path.basename(model_file).replace('.h5', '')
        print(f"🎯 Using model: {model_name}")
        
    def predict_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_best_model() first.")
        
        # Load and preprocess image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)
        
        return prediction

def main():
    parser = argparse.ArgumentParser(description="Iris Classification Predictor")
    parser.add_argument("--image", "-i", type=str, help="Path to single image file")
    parser.add_argument("--model_path", "-m", type=str, default="./results/iris", 
                       help="Path to directory containing trained models")
    
    args = parser.parse_args()
    
    if not args.image:
        print("❌ Please specify --image")
        parser.print_help()
        return
    
    # Initialize predictor
    predictor = IrisPredictor(args.model_path)
    
    try:
        predictor.load_best_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("📋 Make sure you have trained models in the specified directory")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        return
    
    print(f"🔍 Analyzing image: {args.image}")
    try:
        prediction = predictor.predict_image(args.image)
        print(f"📊 Prediction: {prediction[0]}")
        print(f"🎯 Confidence: {np.max(prediction):.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()