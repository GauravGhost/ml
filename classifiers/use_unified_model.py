#!/usr/bin/env python3
"""
Use Unified Multi-Modal Biometric Classifier

This script loads the trained unified model and makes predictions on new images.
"""

import tensorflow as tf
import numpy as np
import json
import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Import centralized GPU utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.gpu_utils import setup_gpu_acceleration

class UnifiedBiometricPredictor:
    def __init__(self):
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.MODEL_PATH = os.path.join(self.PROJECT_ROOT, "results", "unified")
        self.IMG_SIZE = (224, 224)
        
        # Load model and label mapping
        self.load_model_and_labels()
    
    def load_model_and_labels(self):
        """Load the trained model and label mappings"""
        try:
            # Try to load the best model from unified summary
            summary_file = os.path.join(self.MODEL_PATH, 'unified_model_summary.json')
            model_file = None
            
            if os.path.exists(summary_file):
                # Load summary to get best model
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                best_model_name = summary.get('best_model', 'EfficientNetB0')
                model_file = os.path.join(self.MODEL_PATH, f'best_{best_model_name}_model.h5')
                
                if not os.path.exists(model_file):
                    # Try alternative naming
                    model_file = os.path.join(self.MODEL_PATH, f'{best_model_name}_unified_model.h5')
                
                print(f"🏆 Loading best model: {best_model_name}")
            
            # Fallback to old naming scheme
            if not model_file or not os.path.exists(model_file):
                model_file = os.path.join(self.MODEL_PATH, 'unified_model.h5')
                if not os.path.exists(model_file):
                    model_file = os.path.join(self.MODEL_PATH, 'best_model.h5')
            
            if not os.path.exists(model_file):
                # List available models
                available_models = []
                for f in os.listdir(self.MODEL_PATH):
                    if f.endswith('.h5'):
                        available_models.append(f)
                
                if available_models:
                    print(f"Available models: {available_models}")
                    model_file = os.path.join(self.MODEL_PATH, available_models[0])
                    print(f"Loading: {available_models[0]}")
                else:
                    raise FileNotFoundError(f"No model found in {self.MODEL_PATH}")
            
            print(f"Loading model from: {model_file}")
            self.model = tf.keras.models.load_model(model_file)
            
            # Load label mapping
            label_file = os.path.join(self.MODEL_PATH, 'label_mapping.json')
            with open(label_file, 'r') as f:
                self.label_mapping = json.load(f)
            
            # Convert string keys back to integers
            self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
            
            print("✅ Model and labels loaded successfully!")
            print(f"📊 Model can classify {len(self.label_mapping)} different classes")
            print("Available classes:")
            for idx, label in self.label_mapping.items():
                print(f"  {idx}: {label}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Load image
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=self.IMG_SIZE
            )
            
            # Convert to array and normalize
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def predict_image(self, image_path, show_confidence=True):
        """Make prediction on a single image"""
        print(f"🔍 Analyzing: {image_path}")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get class label
        predicted_label = self.label_mapping[predicted_class_idx]
        
        # Parse the prediction
        modality, class_type = predicted_label.split('_', 1)
        
        result = {
            'image_path': image_path,
            'predicted_class': predicted_label,
            'modality': modality,
            'class_type': class_type,
            'confidence': float(confidence),
            'all_predictions': {
                self.label_mapping[i]: float(predictions[0][i]) 
                for i in range(len(predictions[0]))
            }
        }
        
        if show_confidence:
            print(f"🎯 Prediction: {predicted_label}")
            print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"🔬 Modality: {modality}")
            print(f"🏷️  Class: {class_type}")
            
            # Show top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            print("\n🏆 Top 3 predictions:")
            for i, idx in enumerate(top_3_indices):
                label = self.label_mapping[idx]
                conf = predictions[0][idx]
                print(f"  {i+1}. {label}: {conf:.4f} ({conf*100:.2f}%)")
        
        return result
    
    def predict_batch(self, image_paths):
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                result = self.predict_image(image_path, show_confidence=False)
                if result:
                    results.append(result)
                print("-" * 50)
            else:
                print(f"⚠️  Image not found: {image_path}")
        
        return results
    
    def visualize_prediction(self, image_path):
        """Visualize prediction with the image"""
        result = self.predict_image(image_path)
        if not result:
            return
        
        # Load and display image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"Input Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Plot top predictions
        top_5_labels = []
        top_5_confidences = []
        
        sorted_preds = sorted(result['all_predictions'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        for label, conf in sorted_preds:
            top_5_labels.append(label.replace('_', '\n'))
            top_5_confidences.append(conf)
        
        plt.barh(range(len(top_5_labels)), top_5_confidences)
        plt.yticks(range(len(top_5_labels)), top_5_labels)
        plt.xlabel('Confidence')
        plt.title('Top 5 Predictions')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function for interactive prediction"""
    print("🤖 Unified Multi-Modal Biometric Classifier - Prediction Tool")
    print("=" * 60)
    
    # Setup GPU
    setup_gpu_acceleration()
    
    try:
        predictor = UnifiedBiometricPredictor()
        
        while True:
            print("\n" + "=" * 40)
            print("Options:")
            print("1. Predict single image")
            print("2. Predict multiple images")
            print("3. Visualize prediction")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                image_path = input("Enter image path: ").strip()
                if os.path.exists(image_path):
                    predictor.predict_image(image_path)
                else:
                    print(f"❌ Image not found: {image_path}")
            
            elif choice == '2':
                print("Enter image paths (one per line, empty line to finish):")
                image_paths = []
                while True:
                    path = input().strip()
                    if not path:
                        break
                    image_paths.append(path)
                
                if image_paths:
                    results = predictor.predict_batch(image_paths)
                    
                    # Summary
                    print(f"\n📊 Batch Prediction Summary:")
                    print(f"Total images: {len(image_paths)}")
                    print(f"Successfully processed: {len(results)}")
                    
                    # Group by modality
                    modality_counts = {}
                    for result in results:
                        modality = result['modality']
                        modality_counts[modality] = modality_counts.get(modality, 0) + 1
                    
                    print("Results by modality:")
                    for modality, count in modality_counts.items():
                        print(f"  {modality}: {count} images")
                
            elif choice == '3':
                image_path = input("Enter image path for visualization: ").strip()
                if os.path.exists(image_path):
                    predictor.visualize_prediction(image_path)
                else:
                    print(f"❌ Image not found: {image_path}")
            
            elif choice == '4':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()