#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import shutil
from pathlib import Path

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

# SIMPLIFIED Configuration for Better Face Spoof Detection Accuracy
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25  # Match test file for optimal results
INITIAL_EPOCHS = 15  # Initial training like test file
FINE_TUNE_EPOCHS = 10  # Fine-tuning like test file
LEARNING_RATE = 0.001
FINE_TUNE_RATE = 0.0002

# Fix path resolution - go up two levels from classifiers/face/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "face_organized")
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "face")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Using organized dataset: {DATASET_PATH}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Simplified Data Augmentation for Better Learning
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,  # Reduced rotation
    width_shift_range=0.08,  # Reduced shift
    height_shift_range=0.08,
    horizontal_flip=True,
    zoom_range=0.1,  # Reduced zoom
    brightness_range=[0.9, 1.1],  # Reduced brightness variation
    fill_mode='nearest'
)

# Validation data with minimal preprocessing for consistent evaluation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

def fix_face_dataset_labels():
    """
    Automatically fix dataset labeling by moving spoof images from live folder to spoof folder.
    Based on filename patterns that indicate spoofing attacks.
    """
    live_dir = Path(DATASET_PATH) / "live"
    spoof_dir = Path(DATASET_PATH) / "spoof"
    
    # Create directories if they don't exist
    live_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to organize data from scratch
    live_count = len([f for f in live_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    spoof_count = len([f for f in spoof_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if live_count == 0 and spoof_count == 0:
        print("📦 Organizing dataset from original face data...")
        organize_from_original_data()
        return
    
    # Spoof attack patterns in filenames
    spoof_patterns = [
        'bobblehead',
        'resin_projection', 
        'resin',
        'mask',
        'filament',
        'mannequin',
        'HQ_3D_MASK',
        'cloth_mask'
    ]
    
    moved_count = 0
    
    print("🔍 Scanning for misplaced spoof images...")
    
    for file_path in live_dir.glob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            filename_lower = file_path.name.lower()
            
            # Check if filename contains spoof patterns
            for pattern in spoof_patterns:
                if pattern.lower() in filename_lower:
                    # Move to spoof folder
                    destination = spoof_dir / file_path.name
                    print(f"Moving {file_path.name} -> spoof/")
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    break
    
    if moved_count > 0:
        print(f"✅ Dataset fix complete! Moved {moved_count} spoof images from live/ to spoof/")
        
        # Show updated counts
        live_count = len([f for f in live_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        spoof_count = len([f for f in spoof_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        print(f"📊 Updated counts:")
        print(f"   Live: {live_count} images")
        print(f"   Spoof: {spoof_count} images")
    else:
        print("✅ No misplaced images found - dataset already correctly labeled")

def organize_from_original_data():
    """Organize images from original data/face structure into live/spoof folders"""
    original_face_dir = Path(PROJECT_ROOT) / "data" / "face"
    live_dir = Path(DATASET_PATH) / "live"
    spoof_dir = Path(DATASET_PATH) / "spoof"
    
    if not original_face_dir.exists():
        print(f"❌ Original face data not found at {original_face_dir}")
        return
    
    print("📋 Organizing images from original face data structure...")
    
    # Copy live subject images
    live_source = original_face_dir / "live_subject_images"
    if live_source.exists():
        for img_file in live_source.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(img_file, live_dir / img_file.name)
        print(f"📸 Copied {len(list(live_dir.glob('*')))} live images")
    
    # Copy spoof images from various spoof directories
    spoof_patterns = [
        "bobblehead_images", "filament_projection_images", "Full_cloth_mask_images",
        "half_cloth_mask_images", "HQ_3D_MASK_images", "mannequin_projection_images", 
        "resin_projection_images", "white_filament_images", "white_mannequin_images",
        "white_resin_images"
    ]
    
    spoof_count = 0
    for pattern in spoof_patterns:
        pattern_dir = original_face_dir / pattern
        if pattern_dir.exists():
            # Handle device-specific subdirectories
            for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                device_dir = pattern_dir / device
                if device_dir.exists():
                    for img_file in device_dir.glob('*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            new_name = f"{pattern}_{device}_{img_file.name}"
                            shutil.copy2(img_file, spoof_dir / new_name)
                            spoof_count += 1
    
    print(f"👺 Copied {spoof_count} spoof images")
    print(f"✅ Dataset organization complete!")
    print(f"   Live: {len(list(live_dir.glob('*')))} images")
    print(f"   Spoof: {spoof_count} images")

# Automatically fix dataset labels before training
print("🔧 Checking dataset labels...")
fix_face_dataset_labels()

# Skip auto-organization since we're using the corrected face_organized dataset
# DATASET_PATH = organize_face_data_automatically()

if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found at {DATASET_PATH}")
    print("📋 Please update DATASET_PATH variable to point to your face dataset")
    print("   Expected structure: DATASET_PATH/class1/, DATASET_PATH/class2/, etc.")
    exit(1)

class_folders = [d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
num_classes = len(class_folders)

print(f"📂 Found {num_classes} classes: {class_folders}")

if num_classes == 2:
    class_mode = 'binary'
    activation = 'sigmoid'
    loss = 'binary_crossentropy'
    output_units = 1
    print("🎯 Using binary classification")
else:
    class_mode = 'categorical' 
    activation = 'softmax'
    loss = 'categorical_crossentropy'
    output_units = num_classes
    print(f"🎯 Using multi-class classification ({num_classes} classes)")

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
    """Enhanced training function for 90%+ accuracy"""
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

    # Simple but effective architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Just 2 layers - simpler is often better
    x = Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Less dropout
    
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Even less dropout
    
    output = Dense(output_units, activation=activation)(x)

    model = Model(base_model.input, output)

    # Balanced class weighting - less aggressive
    if class_mode == 'binary':
        total_samples = train_data.samples
        class_counts = np.bincount(train_data.classes)
        # Balanced weighting without over-emphasis
        live_weight = total_samples / (2.0 * class_counts[0])
        spoof_weight = total_samples / (2.0 * class_counts[1])
        class_weight = {0: live_weight, 1: spoof_weight}
        print(f"📊 Using balanced class weights: {class_weight}")
        print(f"   Live samples: {class_counts[0]}, Spoof samples: {class_counts[1]}")
    else:
        class_weight = None

    # Simple optimizer
    initial_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=initial_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")
    
    # Simple but effective callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,  # Less patience for quicker training
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,  # Less aggressive reduction
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{SAVE_PATH}/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train top layers only (like test file)
    print(f"\n🚀 Phase 1: Training top layers ({INITIAL_EPOCHS} epochs)")
    history1 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning last 50 layers (like test file)
    print(f"\n🔧 Phase 2: Fine-tuning last 50 layers ({FINE_TUNE_EPOCHS} epochs)")
    
    # Unfreeze last 50 layers exactly like test file
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_RATE),
        loss=loss,
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_data,
        epochs=EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,  # Use same callbacks as test file
        verbose=1
    )
    
    # Combine training histories
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

    print(f"\n🏁 Starting HIGH-ACCURACY face spoof detection training for {len(models)} models...")
    print(f"🎯 Target Accuracy: 80-95% (Optimized Configuration)")
    print(f"📊 Data Balance: ~70% Real, ~30% Spoof (Optimal for Spoof Detection)")
    print(f"⚙️  Enhanced Configuration: {EPOCHS} epochs, batch size {BATCH_SIZE}, image size {IMG_SIZE}")

    # Train each model
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