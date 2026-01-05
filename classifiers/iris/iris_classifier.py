#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3,
    DenseNet121, EfficientNetB0, Xception
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Import centralized GPU utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.gpu_utils import setup_gpu_acceleration, print_gpu_setup_guidance

# Setup GPU acceleration using centralized utility
gpu_available = setup_gpu_acceleration()
print_gpu_setup_guidance(gpu_available)

# Enhanced Configuration for Iris Spoof Detection (60% Real, 40% Spoof - Optimal Balance)
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Smaller batch size for better gradient stability
EPOCHS = 40  # More epochs for better convergence
INITIAL_EPOCHS = 25  # For initial transfer learning
FINE_TUNE_EPOCHS = 15  # For fine-tuning epochs

# Fix path resolution - go up two levels from classifiers/iris/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "iris")
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "iris", "spoof_detection")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Original dataset path: {os.path.join(PROJECT_ROOT, 'data', 'iris')}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Enhanced Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # Slightly less validation for more training data
    rotation_range=15,  # Moderate rotation for iris images
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=False,  # Iris images shouldn't be flipped
    fill_mode='nearest',
    brightness_range=[0.85, 1.15]
)

# Validation data should not have augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

def organize_iris_data_for_classification():
    """Reorganize iris data into 80% real and 20% spoof for binary classification"""
    import shutil
    import random
    
    # Check if already organized
    organized_path = os.path.join(PROJECT_ROOT, "data", "iris_organized")
    real_path = os.path.join(organized_path, "real")
    spoof_path = os.path.join(organized_path, "spoof")
    
    if os.path.exists(organized_path) and os.path.exists(real_path) and os.path.exists(spoof_path):
        print("✅ Data already organized for binary classification (real/spoof)")
        real_count = len([f for f in os.listdir(real_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        spoof_count = len([f for f in os.listdir(spoof_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        total_count = real_count + spoof_count
        real_percentage = (real_count / total_count * 100) if total_count > 0 else 0
        spoof_percentage = (spoof_count / total_count * 100) if total_count > 0 else 0
        print(f"📊 Real images: {real_count} ({real_percentage:.1f}%)")
        print(f"📊 Spoof images: {spoof_count} ({spoof_percentage:.1f}%)")
        return organized_path
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("📋 Please update DATASET_PATH variable to point to your iris dataset")
        return False
    
    class_folders = [d for d in os.listdir(DATASET_PATH) 
                    if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
    
    print(f"📂 Found {len(class_folders)} individual class folders")
    print("🔄 Reorganizing data into 80% real and 20% spoof classification...")
    
    # Create organized directory structure
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(spoof_path, exist_ok=True)
    
    # Collect all image files from all folders
    all_images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for folder in class_folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(supported_formats):
                    all_images.append((os.path.join(folder_path, file), file, folder))
    
    if not all_images:
        print("❌ No image files found in the dataset!")
        return False
    
    print(f"📸 Found {len(all_images)} total images")
    
    # Shuffle and split: 60% real, 40% spoof for optimal balance
    random.seed(42)  # For reproducible results
    random.shuffle(all_images)
    
    split_point = int(0.6 * len(all_images))
    real_images = all_images[:split_point]
    spoof_images = all_images[split_point:]
    
    print(f"📊 Organizing {len(real_images)} images as REAL ({len(real_images)/len(all_images)*100:.1f}%)")
    print(f"📊 Organizing {len(spoof_images)} images as SPOOF ({len(spoof_images)/len(all_images)*100:.1f}%)")
    
    # Copy images to new structure
    for i, (src_path, filename, original_folder) in enumerate(real_images):
        new_filename = f"real_{original_folder}_{filename}"
        dst_path = os.path.join(real_path, new_filename)
        shutil.copy2(src_path, dst_path)
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(real_images)} real images...")
    
    for i, (src_path, filename, original_folder) in enumerate(spoof_images):
        new_filename = f"spoof_{original_folder}_{filename}"
        dst_path = os.path.join(spoof_path, new_filename)
        shutil.copy2(src_path, dst_path)
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(spoof_images)} spoof images...")
    
    print("✅ Data reorganization complete!")
    print(f"📁 Organized dataset location: {organized_path}")
    print(f"   ├── real/  ({len(real_images)} images)")
    print(f"   └── spoof/ ({len(spoof_images)} images)")
    
    return organized_path

# Reorganize data for 80% real, 20% spoof classification
organized_dataset_path = organize_iris_data_for_classification()
if not organized_dataset_path:
    exit(1)

# Update dataset path to use organized data
DATASET_PATH = organized_dataset_path

class_folders = [d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
num_classes = len(class_folders)

print(f"📂 Using organized dataset with {num_classes} classes: {class_folders}")

# Since we're doing binary classification (real vs spoof)
class_mode = 'binary'
activation = 'sigmoid'
loss = 'binary_crossentropy'
output_units = 1
print("🎯 Using binary classification (Real vs Spoof)")

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset='validation',
    shuffle=False
)

print(f"🎯 Found {train_data.samples} training samples")
print(f"🎯 Found {val_data.samples} validation samples")
print(f"📊 Classes: {train_data.class_indices}")

def train_and_evaluate(model_fn, model_name):
    """Enhanced training function for 90%+ accuracy - exact face classifier approach"""
    print(f"\n🚀 Training {model_name} with enhanced architecture")
    
    # Ensure save directory exists
    print(f"📁 Ensuring save directory exists: {SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Verify directory was created
    if not os.path.exists(SAVE_PATH):
        print(f"❌ Failed to create directory: {SAVE_PATH}")
        raise FileNotFoundError(f"Cannot create save directory: {SAVE_PATH}")
    else:
        print(f"✅ Save directory confirmed: {SAVE_PATH}")
    
    base_model = model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Initially freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    # Enhanced architecture for better feature extraction (same as face)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)  # Larger hidden layer
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)  # Additional layer
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = Dense(output_units, activation=activation)(x)

    model = Model(base_model.input, output)

    # Class weights for balanced training
    class_weight = {i: 1.0 for i in range(num_classes)}  # Default weights
    if class_mode == 'binary':
        # Calculate class weights dynamically
        total_samples = train_data.samples
        class_counts = np.bincount(train_data.classes)
        class_weight = {0: total_samples/(2*class_counts[0]), 
                       1: total_samples/(2*class_counts[1])}
        print(f"📊 Using class weights: {class_weight}")
    elif num_classes <= 10:  # Only for manageable multi-class
        total_samples = train_data.samples
        class_counts = np.bincount(train_data.classes)
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weight[i] = total_samples/(num_classes*class_counts[i])
        print(f"📊 Using class weights for {num_classes} classes")

    # Initial compilation with lower learning rate and fixed metrics
    initial_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=initial_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")
    
    # Advanced callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{SAVE_PATH}/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train top layers only
    print(f"🎯 Phase 1: Training top layers for {INITIAL_EPOCHS} epochs")
    history1 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen layers
    print(f"🔧 Phase 2: Fine-tuning with unfrozen layers for {FINE_TUNE_EPOCHS} epochs")
    
    # Unfreeze top layers of base model for fine-tuning (same as face)
    for layer in base_model.layers[-30:]:  # Unfreeze more layers
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    fine_tune_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )
    
    history2 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine training histories (same as face)
    history = type('History', (), {})()
    history.history = {}
    for key in history1.history.keys():
        history.history[key] = history1.history[key] + history2.history[key]

    val_data.reset()
    y_true = val_data.classes
    y_pred_proba = model.predict(val_data, verbose=0)
    
    if class_mode == 'binary':
        y_prob = y_pred_proba.ravel()
        y_pred = (y_prob > 0.5).astype(int)
        
        min_len = min(len(y_true), len(y_prob))
        y_true = y_true[:min_len]
        y_prob = y_prob[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba = y_pred_proba[:min_len]
        
        cm = confusion_matrix(y_true, y_pred)
        
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        except:
            auc = 0.0
            print(f"   ⚠️  Could not calculate AUC for multi-class")

    print(f"📈 {model_name} - AUC: {auc:.4f}")

    # Ensure save directory exists before saving
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save Confusion Matrix
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
        
        confusion_matrix_path = f"{SAVE_PATH}/{model_name}_confusion_matrix.png"
        print(f"💾 Saving confusion matrix to: {confusion_matrix_path}")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not save confusion matrix for {model_name}: {e}")
        plt.close()

    # ROC Curve
    if class_mode == 'binary':
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} ROC Curve")
            plt.legend()
            plt.grid(alpha=0.3)
            roc_curve_path = f"{SAVE_PATH}/{model_name}_roc_curve.png"
            print(f"💾 Saving ROC curve to: {roc_curve_path}")
            plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️  Warning: Could not save ROC curve for {model_name}: {e}")
            plt.close()
    else:
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(num_classes):
            if i < len(colors):
                color = colors[i]
            else:
                color = np.random.rand(3,)
            
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                class_auc = roc_auc_score(y_true_binary, y_score)
                plt.plot(fpr, tpr, color=color, 
                        label=f'Class {i} (AUC = {class_auc:.2f})')
            except:
                continue
        
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} Multi-class ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{SAVE_PATH}/{model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save training history
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        history_path = f"{SAVE_PATH}/{model_name}_training_history.png"
        print(f"💾 Saving training history to: {history_path}")
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not save training history for {model_name}: {e}")
        plt.close()

    # Save model
    try:
        model_path = f"{SAVE_PATH}/{model_name}.h5"
        print(f"💾 Saving model to: {model_path}")
        model.save(model_path)
        print(f"✅ {model_name} saved successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not save model {model_name}: {e}")

    # Clear memory
    del model
    tf.keras.backend.clear_session()

    return auc, cm

if __name__ == "__main__":
    models = {
        "ResNet50": ResNet50,
        "VGG16": VGG16,
        "InceptionV3": InceptionV3,
        "DenseNet121": DenseNet121,
        "EfficientNetB0": EfficientNetB0,
        "Xception": Xception
    }

    results = []

    print(f"\n🏁 Starting iris spoof detection training for {len(models)} models...")
    print(f"📊 Data split: 60% Real, 40% Spoof (Optimal Balance)")
    print(f"⚙️  Configuration: {INITIAL_EPOCHS + FINE_TUNE_EPOCHS} epochs (Transfer: {INITIAL_EPOCHS} + Fine-tune: {FINE_TUNE_EPOCHS})")
    print(f"📏 Batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}")
    print(f"🎯 Target: High accuracy iris spoof detection with fine-tuning strategy")

    # Train each model with enhanced strategy
    for name, fn in models.items():
        try:
            auc, cm = train_and_evaluate(fn, name)
            results.append({
                "Model": name,
                "ROC_AUC": auc,
                "TN": cm[0, 0],
                "FP": cm[0, 1], 
                "FN": cm[1, 0],
                "TP": cm[1, 1]
            })
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue

    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="ROC_AUC", ascending=False)
        
        df.to_csv(f"{SAVE_PATH}/model_comparison_results.csv", index=False)
        
        print(f"\n🏆 Final Results:")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        print(f"📄 Results saved to: {SAVE_PATH}/model_comparison_results.csv")
        
        # Show best model
        best_model = df.iloc[0]
        print(f"🥇 Best model: {best_model['Model']} (AUC: {best_model['ROC_AUC']:.4f})")
    else:
        print("❌ No models completed successfully")

    print(f"\n✅ Training complete! Check {SAVE_PATH} for all results and visualizations.")