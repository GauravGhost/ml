#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import argparse

class FaceAntiSpoofingPredictor:
    def __init__(self, model_path):
        """
        Initialize the face anti-spoofing predictor
        
        Args:
            model_path: Path to the trained model (.h5 file)
        """
        self.model = load_model(model_path)
        self.model_name = os.path.basename(model_path).replace('.h5', '')
        
        # Determine if this is binary or multi-class based on output shape
        output_shape = self.model.output.shape[-1]
        if output_shape == 1:
            self.classification_type = "binary"
            self.classes = ['Live', 'Spoof']
        else:
            self.classification_type = "multiclass"
            # These should match the classes from training
            self.classes = [
                'bobblehead', 'filament_projection', 'full_cloth_mask', 
                'half_cloth_mask', 'hq_3d_mask', 'live', 'mannequin_projection',
                'resin_projection', 'white_filament', 'white_mannequin', 'white_resin'
            ]
        
        print(f"🎭 Loaded {self.model_name} model")
        print(f"📊 Classification type: {self.classification_type}")
        print(f"🎯 Classes: {self.classes}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        return img_array
    
    def predict_single_image(self, image_path):
        """
        Predict on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)[0]
        
        if self.classification_type == "binary":
            # Binary classification
            spoof_confidence = float(prediction[0])
            live_confidence = 1.0 - spoof_confidence
            
            predicted_class = "Spoof" if spoof_confidence > 0.5 else "Live"
            confidence = spoof_confidence if predicted_class == "Spoof" else live_confidence
            
            return {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'live_confidence': live_confidence,
                'spoof_confidence': spoof_confidence,
                'is_live': predicted_class == "Live"
            }
        else:
            # Multi-class classification
            predicted_idx = np.argmax(prediction)
            predicted_class = self.classes[predicted_idx]
            confidence = float(prediction[predicted_idx])
            
            # Create confidence scores for all classes
            class_confidences = {cls: float(conf) for cls, conf in zip(self.classes, prediction)}
            
            return {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_confidences': class_confidences,
                'is_live': predicted_class == "live"
            }
    
    def predict_directory(self, directory_path, output_file=None):
        """
        Predict on all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results as CSV
            
        Returns:
            List of prediction results
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(directory_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"⚠️  No image files found in {directory_path}")
            return []
        
        results = []
        print(f"🔍 Processing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(directory_path, image_file)
            try:
                result = self.predict_single_image(image_path)
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"📊 Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"❌ Error processing {image_file}: {str(e)}")
                continue
        
        # Save results if output file is specified
        if output_file and results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"💾 Results saved to: {output_file}")
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and confidence scores
        
        Args:
            image_path: Path to the image file
            save_path: Optional path to save the visualization
        """
        result = self.predict_single_image(image_path)
        
        # Load image for display
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(img)
        ax1.set_title(f"Input Image\n{os.path.basename(image_path)}")
        ax1.axis('off')
        
        # Show predictions
        if self.classification_type == "binary":
            labels = ['Live', 'Spoof']
            confidences = [result['live_confidence'], result['spoof_confidence']]
            colors = ['green' if result['is_live'] else 'red', 'red' if not result['is_live'] else 'green']
            
            bars = ax2.bar(labels, confidences, color=colors, alpha=0.7)
            ax2.set_ylabel('Confidence')
            ax2.set_title(f'Prediction: {result["predicted_class"]}\n'
                         f'Confidence: {result["confidence"]:.3f}')
            ax2.set_ylim(0, 1)
            
            # Add confidence text on bars
            for bar, conf in zip(bars, confidences):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom')
        
        else:
            # Multi-class visualization (show top 5 predictions)
            class_confs = result['class_confidences']
            sorted_classes = sorted(class_confs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            labels = [cls for cls, _ in sorted_classes]
            confidences = [conf for _, conf in sorted_classes]
            colors = ['green' if cls == 'live' else 'orange' if cls == result['predicted_class'] else 'lightblue' 
                     for cls in labels]
            
            bars = ax2.barh(labels, confidences, color=colors, alpha=0.7)
            ax2.set_xlabel('Confidence')
            ax2.set_title(f'Prediction: {result["predicted_class"]}\n'
                         f'Confidence: {result["confidence"]:.3f}')
            ax2.set_xlim(0, 1)
            
            # Add confidence text on bars
            for bar, conf in zip(bars, confidences):
                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{conf:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Face Anti-Spoofing Prediction")
    parser.add_argument("--model", required=True, help="Path to trained model (.h5 file)")
    parser.add_argument("--image", help="Path to single image for prediction")
    parser.add_argument("--directory", help="Path to directory containing images")
    parser.add_argument("--output", help="Output CSV file for batch predictions")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument("--save_viz", help="Path to save visualization (when using --visualize)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Initialize predictor
    predictor = FaceAntiSpoofingPredictor(args.model)
    
    if args.image:
        # Single image prediction
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        result = predictor.predict_single_image(args.image)
        
        print("\n🎭 Prediction Result:")
        print("=" * 40)
        print(f"Image: {os.path.basename(args.image)}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Is Live: {result['is_live']}")
        
        if predictor.classification_type == "binary":
            print(f"Live Confidence: {result['live_confidence']:.4f}")
            print(f"Spoof Confidence: {result['spoof_confidence']:.4f}")
        else:
            print("\nTop 3 Class Confidences:")
            sorted_classes = sorted(result['class_confidences'].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            for cls, conf in sorted_classes:
                print(f"  {cls}: {conf:.4f}")
        
        if args.visualize:
            predictor.visualize_prediction(args.image, args.save_viz)
    
    elif args.directory:
        # Directory prediction
        results = predictor.predict_directory(args.directory, args.output)
        
        if results:
            print(f"\n📊 Batch Prediction Summary:")
            print("=" * 40)
            print(f"Total images processed: {len(results)}")
            
            if predictor.classification_type == "binary":
                live_count = sum(1 for r in results if r['is_live'])
                spoof_count = len(results) - live_count
                print(f"Live predictions: {live_count}")
                print(f"Spoof predictions: {spoof_count}")
                
                avg_live_conf = np.mean([r['live_confidence'] for r in results])
                avg_spoof_conf = np.mean([r['spoof_confidence'] for r in results])
                print(f"Average live confidence: {avg_live_conf:.4f}")
                print(f"Average spoof confidence: {avg_spoof_conf:.4f}")
            else:
                class_counts = {}
                for result in results:
                    cls = result['predicted_class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                print("Class distribution:")
                for cls, count in sorted(class_counts.items()):
                    print(f"  {cls}: {count}")
                
                avg_confidence = np.mean([r['confidence'] for r in results])
                print(f"Average confidence: {avg_confidence:.4f}")
    
    else:
        print("❌ Please specify either --image or --directory")
        parser.print_help()

if __name__ == "__main__":
    main()