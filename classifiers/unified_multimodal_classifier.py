#!/usr/bin/env python3
"""
Unified Multi-Modal Biometric Classifier

This classifier trains on face, fingerprint, and iris datasets simultaneously
using a multi-modal neural network architecture.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3,
    DenseNet121, EfficientNetB0, Xception
)

# Import centralized GPU utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.gpu_utils import setup_gpu_acceleration, print_gpu_setup_guidance

# Setup GPU acceleration
gpu_available = setup_gpu_acceleration()
print_gpu_setup_guidance(gpu_available)

class UnifiedBiometricClassifier:
    def __init__(self, model_name=None):
        # Configuration
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = 30
        self.LEARNING_RATE = 0.001
        self.model_name = model_name
        
        # Paths
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.FACE_PATH = os.path.join(self.PROJECT_ROOT, "data", "face_organized")
        self.FINGERPRINT_PATH = os.path.join(self.PROJECT_ROOT, "data", "fingerprint")
        self.IRIS_PATH = os.path.join(self.PROJECT_ROOT, "data", "iris")
        self.SAVE_PATH = os.path.join(self.PROJECT_ROOT, "results", "unified")
        
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        
        # Dataset info
        self.face_classes = ['live', 'spoof']
        self.fingerprint_classes = ['Altered', 'Real', 'SOCOFing']
        self.iris_classes = []  # Will be populated dynamically
        
        # Label encoders for each modality
        self.face_encoder = LabelEncoder()
        self.fingerprint_encoder = LabelEncoder()
        self.iris_encoder = LabelEncoder()
        
        # Combined label encoder for the unified model
        self.unified_encoder = LabelEncoder()
        
        print(f"📁 Project root: {self.PROJECT_ROOT}")
        print(f"💾 Results will be saved to: {self.SAVE_PATH}")
    
    def load_and_preprocess_images(self, directory, target_size=(224, 224)):
        """Load and preprocess images from directory"""
        images = []
        labels = []
        
        if not os.path.exists(directory):
            print(f"⚠️  Directory not found: {directory}")
            return np.array([]), np.array([])
        
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if not os.path.isdir(class_path):
                continue
                
            print(f"Processing {class_folder} from {directory}...")
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Load and preprocess image
                        img = tf.keras.preprocessing.image.load_img(
                            img_path, target_size=target_size
                        )
                        img_array = tf.keras.preprocessing.image.img_to_array(img)
                        img_array = img_array / 255.0  # Normalize
                        
                        images.append(img_array)
                        labels.append(class_folder)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def prepare_datasets(self):
        """Prepare and combine all three datasets"""
        print("🔄 Loading datasets...")
        
        # Load Face Dataset
        print("Loading face dataset...")
        face_images, face_labels = self.load_and_preprocess_images(self.FACE_PATH)
        face_labels_prefixed = np.array([f"face_{label}" for label in face_labels])
        
        # Load Fingerprint Dataset  
        print("Loading fingerprint dataset...")
        fingerprint_images, fingerprint_labels = self.load_and_preprocess_images(self.FINGERPRINT_PATH)
        fingerprint_labels_prefixed = np.array([f"fingerprint_{label}" for label in fingerprint_labels])
        
        # Load Iris Dataset (limit classes for manageable training)
        print("Loading iris dataset...")
        iris_dirs = [d for d in os.listdir(self.IRIS_PATH) if os.path.isdir(os.path.join(self.IRIS_PATH, d))]
        iris_dirs = sorted(iris_dirs)[:20]  # Limit to first 20 classes for manageable training
        
        iris_images_list = []
        iris_labels_list = []
        
        for iris_class in iris_dirs:
            iris_class_path = os.path.join(self.IRIS_PATH, iris_class)
            class_images, class_labels = self.load_and_preprocess_images(iris_class_path)
            if len(class_images) > 0:
                iris_images_list.extend(class_images)
                iris_labels_list.extend([f"iris_{iris_class}" for _ in class_labels])
        
        iris_images = np.array(iris_images_list)
        iris_labels = np.array(iris_labels_list)
        
        # Combine all datasets
        print("Combining datasets...")
        all_images = []
        all_labels = []
        
        if len(face_images) > 0:
            all_images.extend(face_images)
            all_labels.extend(face_labels_prefixed)
            print(f"✅ Face: {len(face_images)} samples")
        
        if len(fingerprint_images) > 0:
            all_images.extend(fingerprint_images)
            all_labels.extend(fingerprint_labels_prefixed)
            print(f"✅ Fingerprint: {len(fingerprint_images)} samples")
            
        if len(iris_images) > 0:
            all_images.extend(iris_images)
            all_labels.extend(iris_labels)
            print(f"✅ Iris: {len(iris_images)} samples")
        
        if not all_images:
            raise ValueError("No images found in any dataset!")
        
        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        
        # Encode labels
        y_encoded = self.unified_encoder.fit_transform(y)
        
        # Save label mapping
        label_mapping = {i: label for i, label in enumerate(self.unified_encoder.classes_)}
        with open(os.path.join(self.SAVE_PATH, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"📊 Total samples: {len(X)}")
        print(f"📊 Total classes: {len(np.unique(y))}")
        print(f"📊 Image shape: {X[0].shape}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\n📊 Class Distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count} samples")
        
        return X, y_encoded
    
    def create_model(self, num_classes, base_model_fn=EfficientNetB0, model_name="EfficientNetB0"):
        """Create unified multi-modal model"""
        print(f"🏗️ Creating {model_name} model for {num_classes} classes...")
        
        # Base model
        base_model = base_model_fn(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.IMG_SIZE, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom top layers
        inputs = tf.keras.Input(shape=(*self.IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Multi-modal feature extraction layers
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_single_model(self, X, y, base_model_fn, model_name):
        """Train a single model and return results"""
        print(f"🚀 Starting training for {model_name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        num_classes = len(np.unique(y))
        model = self.create_model(num_classes, base_model_fn, model_name)
        
        print(f"\n📋 {model_name} Architecture Summary:")
        model.summary()
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.SAVE_PATH, f'best_{model_name}_model.h5'),
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.BATCH_SIZE),
            steps_per_epoch=len(X_train) // self.BATCH_SIZE,
            validation_data=(X_val, y_val),
            epochs=self.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(self.SAVE_PATH, f'{model_name}_unified_model.h5'))
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions for detailed analysis
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
        del model
        
        return test_accuracy, y_test, y_pred_classes, history
    
    def save_comparison_results(self, results):
        """Save comparison results for all trained models"""
        print("💾 Saving comparison results...")
        
        if not results:
            print("❌ No results to save")
            return
        
        # Save individual model results
        for result in results:
            model_name = result["Model"]
            
            # Plot training curves for each model
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(result["history"].history['loss'], label='Training Loss')
            plt.plot(result["history"].history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(result["history"].history['accuracy'], label='Training Accuracy')
            plt.plot(result["history"].history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            # Confusion matrix
            cm = confusion_matrix(result["y_test"], result["y_pred"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.SAVE_PATH, f'{model_name}_training_results.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Classification report for individual model
            class_names = [self.unified_encoder.classes_[i] for i in range(len(self.unified_encoder.classes_))]
            report = classification_report(result["y_test"], result["y_pred"], target_names=class_names, output_dict=True)
            
            with open(os.path.join(self.SAVE_PATH, f'{model_name}_classification_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
        
        # Create comparison plot
        self.create_comparison_plots(results)
        
        # Find best model
        best_model = max(results, key=lambda x: x["Accuracy"])
        best_accuracy = best_model["Accuracy"]
        
        # Save overall summary
        summary = {
            'best_model': best_model["Model"],
            'best_accuracy': float(best_accuracy),
            'total_classes': len(self.unified_encoder.classes_),
            'classes': list(self.unified_encoder.classes_),
            'models_trained': [r["Model"] for r in results],
            'model_accuracies': {r["Model"]: float(r["Accuracy"]) for r in results},
            'best_model_path': os.path.join(self.SAVE_PATH, f'best_{best_model["Model"]}_model.h5'),
            'training_completed': True
        }
        
        with open(os.path.join(self.SAVE_PATH, 'unified_model_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV summary for compatibility with analysis script
        self.save_csv_summary(results)
        
        print(f"✅ All results saved to {self.SAVE_PATH}")
        print(f"🎯 Best Model: {best_model['Model']} with accuracy: {best_accuracy:.4f}")
        
        # Print final comparison
        print(f"\n📊 FINAL MODEL COMPARISON:")
        print("-" * 40)
        for result in sorted(results, key=lambda x: x["Accuracy"], reverse=True):
            print(f"  {result['Model']:<15}: {result['Accuracy']:.4f} ({result['Accuracy']*100:.2f}%)")
    
    def save_csv_summary(self, results):
        """Save CSV summary for compatibility with analysis scripts"""
        if not results:
            return
        
        # Create DataFrame with model results
        csv_data = []
        for result in results:
            csv_data.append({
                'Model': result['Model'],
                'Accuracy': result['Accuracy'],
                'Accuracy_Score': result['Accuracy']
            })
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.SAVE_PATH, 'unified_model_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"📊 CSV summary saved to: {csv_path}")
    
    def create_comparison_plots(self, results):
        """Create comprehensive comparison plots for all models"""
        if not results:
            return
        
        # Extract data for plotting
        models = [r["Model"] for r in results]
        accuracies = [r["Accuracy"] for r in results]
        
        # Create comprehensive comparison plot
        plt.figure(figsize=(20, 12))
        
        # Test accuracy comparison
        plt.subplot(2, 3, 1)
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Training loss comparison (last epoch)
        plt.subplot(2, 3, 2)
        final_losses = [r["history"].history['loss'][-1] for r in results]
        bars = plt.bar(models, final_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        plt.title('Final Training Loss Comparison')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        
        # Validation loss comparison (last epoch)
        plt.subplot(2, 3, 3)
        final_val_losses = [r["history"].history['val_loss'][-1] for r in results]
        bars = plt.bar(models, final_val_losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
        plt.title('Final Validation Loss Comparison')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        
        # Training curves comparison - Accuracy
        plt.subplot(2, 3, 4)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        for i, result in enumerate(results):
            plt.plot(result["history"].history['val_accuracy'], 
                    label=f'{result["Model"]}', color=colors[i % len(colors)])
        plt.title('Validation Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        
        # Training curves comparison - Loss
        plt.subplot(2, 3, 5)
        for i, result in enumerate(results):
            plt.plot(result["history"].history['val_loss'], 
                    label=f'{result["Model"]}', color=colors[i % len(colors)])
        plt.title('Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        # Model ranking
        plt.subplot(2, 3, 6)
        sorted_results = sorted(results, key=lambda x: x["Accuracy"], reverse=True)
        ranks = list(range(1, len(sorted_results) + 1))
        model_names = [r["Model"] for r in sorted_results]
        accuracies = [r["Accuracy"] for r in sorted_results]
        
        bars = plt.barh(ranks, accuracies, color=['#FFD700', '#C0C0C0', '#CD7F32', '#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.title('Model Ranking by Accuracy')
        plt.xlabel('Accuracy')
        plt.ylabel('Rank')
        plt.yticks(ranks, [f'{r}. {name}' for r, name in zip(ranks, model_names)])
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.SAVE_PATH, 'unified_models_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to: {os.path.join(self.SAVE_PATH, 'unified_models_comparison.png')}")
    
    def train_model(self, X, y):
        """Train model(s) based on configuration"""
        print("🚀 Starting unified multi-modal training...")
        
        # Available models
        models = {
            "ResNet50": ResNet50,
            "VGG16": VGG16,
            "InceptionV3": InceptionV3,
            "DenseNet121": DenseNet121,
            "EfficientNetB0": EfficientNetB0,
            "Xception": Xception
        }
        
        results = []
        
        if self.model_name:
            # Train specific model
            if self.model_name not in models:
                raise ValueError(f"Model {self.model_name} not found. Available: {list(models.keys())}")
            
            print(f"🎯 Training specific model: {self.model_name}")
            model_fn = models[self.model_name]
            accuracy, y_test, y_pred, history = self.train_single_model(X, y, model_fn, self.model_name)
            
            results.append({
                "Model": self.model_name,
                "Accuracy": accuracy,
                "y_test": y_test,
                "y_pred": y_pred,
                "history": history
            })
            
        else:
            # Train all models
            print(f"🎯 Training ALL models: {list(models.keys())}")
            print(f"⚠️  This will take approximately {len(models) * self.EPOCHS} total epochs...")
            
            for model_name, model_fn in models.items():
                try:
                    print(f"\n{'='*20} {model_name} {'='*20}")
                    accuracy, y_test, y_pred, history = self.train_single_model(X, y, model_fn, model_name)
                    
                    results.append({
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "y_test": y_test,
                        "y_pred": y_pred,
                        "history": history
                    })
                    
                    print(f"✅ {model_name} completed with accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}")
                    continue
        
        # Save comparison results
        self.save_comparison_results(results)
        
        return results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Biometric Classifier")
    parser.add_argument("--model", "-m", 
                       choices=["ResNet50", "VGG16", "InceptionV3", "DenseNet121", "EfficientNetB0", "Xception"],
                       help="Specific model to train (if not specified, trains all models)")
    
    args = parser.parse_args()
    
    print("🤖 Unified Multi-Modal Biometric Classifier")
    print("=" * 50)
    
    if args.model:
        print(f"🎯 Training specific model: {args.model}")
    else:
        print("🎯 Training ALL 6 models (ResNet50, VGG16, InceptionV3, DenseNet121, EfficientNetB0, Xception)")
        print("⚠️  This will take significantly longer but will compare all architectures")
    
    try:
        # Initialize classifier
        classifier = UnifiedBiometricClassifier(model_name=args.model)
        
        # Prepare datasets
        X, y = classifier.prepare_datasets()
        
        # Train model(s)
        results = classifier.train_model(X, y)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {classifier.SAVE_PATH}")
        
        if len(results) > 1:
            best_result = max(results, key=lambda x: x["Accuracy"])
            print(f"🏆 Best model: {best_result['Model']} with accuracy: {best_result['Accuracy']:.4f}")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()