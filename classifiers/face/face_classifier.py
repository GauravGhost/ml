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

# Enhanced Configuration for 90% Accuracy
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Smaller batch size for better gradient stability
EPOCHS = 50  # More epochs for better convergence
INITIAL_EPOCHS = 25  # For initial transfer learning
FINE_TUNE_EPOCHS = 25  # For fine-tuning epochs

# Fix path resolution - go up two levels from classifiers/face/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "face")
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "face")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Dataset path: {DATASET_PATH}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Enhanced Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # Slightly less validation for more training data
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1
)

# Validation data should not have augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

def organize_face_data_automatically():
    """Automatically organize face data into binary classification if needed"""
    organized_path = os.path.join(PROJECT_ROOT, "data", "face_organized")
    
    # Check if organized data already exists
    if os.path.exists(organized_path) and len(os.listdir(organized_path)) > 0:
        print(f"✅ Using existing organized data: {organized_path}")
        return organized_path
    
    # Check if raw data exists and needs organization
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return DATASET_PATH
    
    # Check if data is already in proper format (live/spoof folders)
    folders = [d for d in os.listdir(DATASET_PATH) 
               if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
    
    print(f"📂 Found folders in dataset: {folders}")
    
    if set(folders) == {'live', 'spoof'} or len(folders) <= 3:
        print(f"✅ Data already properly organized: {DATASET_PATH}")
        return DATASET_PATH
    
    # Auto-organize data for binary classification
    print("🔧 Auto-organizing face data for binary classification...")
    
    if os.path.exists(organized_path):
        import shutil
        shutil.rmtree(organized_path)
    os.makedirs(organized_path, exist_ok=True)
    
    # Create binary classification structure
    live_path = os.path.join(organized_path, "live")
    spoof_path = os.path.join(organized_path, "spoof")
    os.makedirs(live_path, exist_ok=True)
    os.makedirs(spoof_path, exist_ok=True)
    
    # Copy live subjects
    live_source = os.path.join(DATASET_PATH, "live_subject_images")
    live_count = 0
    if os.path.exists(live_source):
        print(f"📸 Processing live subjects from: {live_source}")
        import shutil
        for file in os.listdir(live_source):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                shutil.copy2(
                    os.path.join(live_source, file),
                    os.path.join(live_path, file)
                )
                live_count += 1
    
    # Copy all spoof categories
    spoof_categories = [
        "bobblehead_images", "filament_projection_images", "Full_cloth_mask_images",
        "half_cloth_mask_images", "HQ_3D_MASK_images", "mannequin_projection_images",
        "resin_projection_images", "white_filament_images", "white_mannequin_images",
        "white_resin_images"
    ]
    
    spoof_count = 0
    import shutil
    for category in spoof_categories:
        category_path = os.path.join(DATASET_PATH, category)
        if os.path.exists(category_path):
            print(f"📂 Processing {category}...")
            # Handle both direct images and device-specific folders
            if any(d in os.listdir(category_path) for d in ["DSLR", "IPHONE14", "SAMSUNG_S9"]):
                # Device-specific folders
                for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                    device_path = os.path.join(category_path, device)
                    if os.path.exists(device_path):
                        for file in os.listdir(device_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                shutil.copy2(
                                    os.path.join(device_path, file),
                                    os.path.join(spoof_path, f"{category}_{device}_{file}")
                                )
                                spoof_count += 1
            else:
                # Direct images
                for file in os.listdir(category_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        shutil.copy2(
                            os.path.join(category_path, file),
                            os.path.join(spoof_path, f"{category}_{file}")
                        )
                        spoof_count += 1
    
    print(f"✅ Data organized: {live_count} live, {spoof_count} spoof samples")
    return organized_path

# Auto-organize data and set path
DATASET_PATH = organize_face_data_automatically()

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
    
    base_model = model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Initially freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    # Enhanced architecture for better feature extraction
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
    class_weight = {0: 1.0, 1: 1.0}  # Default weights
    if class_mode == 'binary':
        # Calculate class weights dynamically
        total_samples = train_data.samples
        class_counts = np.bincount(train_data.classes)
        class_weight = {0: total_samples/(2*class_counts[0]), 
                       1: total_samples/(2*class_counts[1])}
        print(f"📊 Using class weights: {class_weight}")

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
    
    # Unfreeze top layers of base model for fine-tuning
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

    # Save Confusion Matrix
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
    
    plt.savefig(f"{SAVE_PATH}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve
    if class_mode == 'binary':
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{SAVE_PATH}/{model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{SAVE_PATH}/{model_name}_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save model
    model.save(f"{SAVE_PATH}/{model_name}.h5")
    print(f"💾 {model_name} saved to {SAVE_PATH}/{model_name}.h5")

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

    print(f"\n🏁 Starting training for {len(models)} models...")
    print(f"⚙️  Configuration: {EPOCHS} epochs, batch size {BATCH_SIZE}, image size {IMG_SIZE}")

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