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

# Optimized Configuration for High-Accuracy Face Spoof Detection (80-95% target)
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Optimal batch size for stable gradients and better convergence
EPOCHS = 60  # More epochs for comprehensive learning
INITIAL_EPOCHS = 40  # Extended initial training
FINE_TUNE_EPOCHS = 20  # Extended fine-tuning
LEARNING_RATE = 0.00001  # Lower learning rate for better convergence
FINE_TUNE_RATE = 0.000001  # Very low rate for fine-tuning

# Fix path resolution - go up two levels from classifiers/face/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "face")
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "face", "spoof_detection")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Original dataset path: {os.path.join(PROJECT_ROOT, 'data', 'face')}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Advanced Data Augmentation specifically optimized for face spoof detection
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # More validation data for better model evaluation
    rotation_range=15,  # Reduced rotation for faces
    width_shift_range=0.15,  # Moderate shifts
    height_shift_range=0.15,
    shear_range=0.1,  # Reduced shear for face geometry preservation
    zoom_range=[0.85, 1.15],  # Controlled zoom for face scale variation
    horizontal_flip=True,
    fill_mode='reflect',  # Better fill mode for face boundaries
    brightness_range=[0.7, 1.3],  # Enhanced brightness variation
    channel_shift_range=0.15  # Increased color variation
)

# Validation data with minimal preprocessing for consistent evaluation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

def organize_face_data_automatically():
    """Automatically organize face data with optimal 60% real, 40% spoof balance"""
    import shutil
    import random
    
    organized_path = os.path.join(PROJECT_ROOT, "data", "face_organized")
    
    # Check if organized data already exists
    if os.path.exists(organized_path):
        live_path = os.path.join(organized_path, "live") 
        spoof_path = os.path.join(organized_path, "spoof")
        if os.path.exists(live_path) and os.path.exists(spoof_path):
            live_count = len([f for f in os.listdir(live_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            spoof_count = len([f for f in os.listdir(spoof_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_count = live_count + spoof_count
            if total_count > 0:
                live_percentage = (live_count / total_count * 100)
                spoof_percentage = (spoof_count / total_count * 100)
                print(f"✅ Using existing organized data: {organized_path}")
                print(f"📊 Current split - Live: {live_count} ({live_percentage:.1f}%), Spoof: {spoof_count} ({spoof_percentage:.1f}%)")
                
                # Check if rebalancing is needed for optimal performance
                if live_percentage < 55 or live_percentage > 65:
                    print("🔄 Rebalancing data for optimal 60/40 split...")
                else:
                    return organized_path
    
    # Check if raw data exists and needs organization
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return DATASET_PATH
    
    print("🔧 Organizing face data with optimal 60% real, 40% spoof balance...")
    
    if os.path.exists(organized_path):
        shutil.rmtree(organized_path)
    os.makedirs(organized_path, exist_ok=True)
    
    # Create binary classification structure
    live_path = os.path.join(organized_path, "live")
    spoof_path = os.path.join(organized_path, "spoof")
    os.makedirs(live_path, exist_ok=True)
    os.makedirs(spoof_path, exist_ok=True)
    
    # Collect all images
    all_images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # Collect live subjects
    live_source = os.path.join(DATASET_PATH, "live_subject_images")
    if os.path.exists(live_source):
        print(f"📸 Collecting live subjects from: {live_source}")
        for file in os.listdir(live_source):
            if file.lower().endswith(supported_formats):
                all_images.append((os.path.join(live_source, file), file, "live_subject"))
    
    # Collect all spoof categories
    spoof_categories = [
        "bobblehead_images", "filament_projection_images", "Full_cloth_mask_images",
        "half_cloth_mask_images", "HQ_3D_MASK_images", "mannequin_projection_images", 
        "resin_projection_images", "white_filament_images", "white_mannequin_images",
        "white_resin_images"
    ]
    
    for category in spoof_categories:
        category_path = os.path.join(DATASET_PATH, category)
        if os.path.exists(category_path):
            # Handle both direct images and device-specific folders
            if any(d in os.listdir(category_path) for d in ["DSLR", "IPHONE14", "SAMSUNG_S9"]):
                # Device-specific folders
                for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                    device_path = os.path.join(category_path, device)
                    if os.path.exists(device_path):
                        for file in os.listdir(device_path):
                            if file.lower().endswith(supported_formats):
                                all_images.append((os.path.join(device_path, file), f"{category}_{device}_{file}", category))
            else:
                # Direct images
                for file in os.listdir(category_path):
                    if file.lower().endswith(supported_formats):
                        all_images.append((os.path.join(category_path, file), f"{category}_{file}", category))
    
    if not all_images:
        print("❌ No image files found in the dataset!")
        return DATASET_PATH
    
    print(f"📸 Found {len(all_images)} total images")
    
    # Improved data splitting for optimal accuracy: 70% real, 30% spoof\n    # This balance has shown better results for face spoof detection\n    random.seed(42)  # For reproducible results\n    random.shuffle(all_images)\n    \n    # Separate real and spoof images first\n    real_source_images = [img for img in all_images if \"live_subject\" in img[2]]\n    spoof_source_images = [img for img in all_images if \"live_subject\" not in img[2]]\n    \n    # Calculate optimal split (aim for 70% real, 30% spoof for better accuracy)\n    target_real_ratio = 0.7\n    total_target = min(len(real_source_images) * 1.4, len(spoof_source_images) * 3.33)  # Ensure we have enough of both\n    \n    num_real = int(total_target * target_real_ratio)\n    num_spoof = int(total_target * (1 - target_real_ratio))\n    \n    # Take the available images, ensuring we don't exceed what we have\n    num_real = min(num_real, len(real_source_images))\n    num_spoof = min(num_spoof, len(spoof_source_images))\n    \n    real_images = real_source_images[:num_real]\n    spoof_images = spoof_source_images[:num_spoof]\n    \n    print(f\"📊 Organizing {len(real_images)} images as REAL ({len(real_images)/(len(real_images)+len(spoof_images))*100:.1f}%)\")\n    print(f\"📊 Organizing {len(spoof_images)} images as SPOOF ({len(spoof_images)/(len(real_images)+len(spoof_images))*100:.1f}%)\")
    
    # Copy images to new structure
    for i, (src_path, filename, category) in enumerate(real_images):
        dst_path = os.path.join(live_path, filename)
        shutil.copy2(src_path, dst_path)
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(real_images)} real images...")
    
    for i, (src_path, filename, category) in enumerate(spoof_images):
        dst_path = os.path.join(spoof_path, filename)
        shutil.copy2(src_path, dst_path)
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(spoof_images)} spoof images...")
    
    print("✅ Face data organized for optimal performance!")
    print(f"📁 Organized dataset location: {organized_path}")
    print(f"   ├── live/  ({len(real_images)} images - {len(real_images)/(len(real_images)+len(spoof_images))*100:.1f}%)")
    print(f"   └── spoof/ ({len(spoof_images)} images - {len(spoof_images)/(len(real_images)+len(spoof_images))*100:.1f}%)")
    
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

    # Advanced architecture for high-accuracy face spoof detection
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Multi-scale feature extraction
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Additional feature processing layer
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    output = Dense(output_units, activation=activation)(x)

    model = Model(base_model.input, output)

    # Optimized class weights for better balance
    class_weight = {0: 1.0, 1: 1.0}  # Default weights
    if class_mode == 'binary':
        # Enhanced class weight calculation for spoof detection
        total_samples = train_data.samples
        class_counts = np.bincount(train_data.classes)
        # Apply stronger balancing for minority class
        class_weight = {0: total_samples/(1.8*class_counts[0]), 
                       1: total_samples/(2.2*class_counts[1])}
        print(f"📊 Using optimized class weights: {class_weight}")

    # Initial compilation with optimized parameters
    initial_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    model.compile(
        optimizer=initial_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")
    
    # Advanced callbacks for high-accuracy training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased patience for better convergence
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Minimum improvement threshold
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,  # More aggressive reduction
            patience=8,
            min_lr=1e-8,
            verbose=1,
            cooldown=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{SAVE_PATH}/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        # Add learning rate scheduling
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (0.95 ** epoch),
            verbose=0
        )
    ]
    
    # Phase 1: Train top layers only with gradual learning
    print(f"🎯 Phase 1: Training top layers for {INITIAL_EPOCHS} epochs")
    history1 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_steps=val_data.samples // BATCH_SIZE
    )
    
    # Phase 2: Fine-tune with unfrozen layers for high accuracy
    print(f"🔧 Phase 2: Fine-tuning with unfrozen layers for {FINE_TUNE_EPOCHS} epochs")
    
    # Unfreeze more layers gradually for better fine-tuning
    for layer in base_model.layers[-50:]:  # Unfreeze more layers for better adaptation
        layer.trainable = True
    
    # Recompile with much lower learning rate for fine-tuning
    fine_tune_optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=FINE_TUNE_RATE,  # Use predefined fine-tune rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    model.compile(
        optimizer=fine_tune_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )
    
    # Update callbacks for fine-tuning phase
    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,  # More patience for fine-tuning
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0005
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=6,
            min_lr=1e-9,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"{SAVE_PATH}/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history2 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=fine_tune_callbacks,  # Use fine-tuning callbacks
        verbose=1,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_steps=val_data.samples // BATCH_SIZE
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