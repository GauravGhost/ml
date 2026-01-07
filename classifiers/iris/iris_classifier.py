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

# SIMPLIFIED Configuration for Better Iris PERSON RECOGNITION Accuracy - EXACT FACE CLASSIFIER CONFIG
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Match face classifier batch size
EPOCHS = 25  # Match face classifier total epochs
INITIAL_EPOCHS = 15  # Initial training like face classifier
FINE_TUNE_EPOCHS = 10  # Fine-tuning like face classifier
LEARNING_RATE = 0.001  # Match face classifier learning rate
FINE_TUNE_RATE = 0.0002  # Match face classifier fine-tune rate

# Use subset of iris classes for better accuracy (too many classes = lower accuracy)
MAX_CLASSES = 50  # Limit to 50 people for high accuracy

# Fix path resolution - go up two levels from classifiers/iris/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "iris")
SAVE_PATH = os.path.join(PROJECT_ROOT, "results", "iris", "person_recognition")

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Original dataset path: {os.path.join(PROJECT_ROOT, 'data', 'iris')}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Simplified Data Augmentation for Better Learning - OPTIMIZED FOR IRIS
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=5,  # Minimal rotation for iris
    width_shift_range=0.05,  # Minimal shift for iris
    height_shift_range=0.05,
    horizontal_flip=False,  # Don't flip iris images
    zoom_range=0.05,  # Minimal zoom for iris
    brightness_range=[0.95, 1.05],  # Minimal brightness for iris
    fill_mode='nearest'
)

# Validation data with minimal preprocessing for consistent evaluation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

def prepare_iris_person_recognition():
    """Select subset of iris classes for high accuracy person recognition"""
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return False
    
    # Get all available class folders
    all_classes = [d for d in os.listdir(DATASET_PATH) 
                   if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
    
    print(f"📂 Found {len(all_classes)} total iris classes")
    
    # Filter classes with sufficient images
    valid_classes = []
    for class_name in all_classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        image_count = len([f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        if image_count >= 10:  # Need at least 10 images per person
            valid_classes.append((class_name, image_count))
    
    # Sort by image count and take top classes
    valid_classes.sort(key=lambda x: x[1], reverse=True)
    selected_classes = valid_classes[:MAX_CLASSES]
    
    print(f"📊 Using {len(selected_classes)} people for iris recognition:")
    total_images = sum([count for _, count in selected_classes])
    print(f"   Total images: {total_images}")
    print(f"   Average per person: {total_images/len(selected_classes):.1f}")
    
    return DATASET_PATH

# Prepare iris person recognition dataset
organized_dataset_path = prepare_iris_person_recognition()
if not organized_dataset_path:
    exit(1)

class_folders = [d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
num_classes = len(class_folders)

print(f"📂 Using {num_classes} iris classes for person recognition")

# Multi-class iris person recognition
class_mode = 'categorical'
activation = 'softmax'
loss = 'categorical_crossentropy'
output_units = num_classes
print(f"🎯 Using multi-class iris person recognition ({num_classes} people)")

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

    # Simple but effective architecture - EXACT FACE CLASSIFIER ARCHITECTURE
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

    # No class weights needed for balanced multi-class iris recognition
    class_weight = None

    # Simple optimizer - EXACT FACE CLASSIFIER CONFIG
    initial_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=initial_optimizer,
        loss=loss,
        metrics=['accuracy']  # Keep only basic accuracy for compatibility
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")
    
    # Simple but effective callbacks - EXACT FACE CLASSIFIER CONFIG
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
    
    # Phase 1: Train top layers only (like face classifier)
    print(f"\n🚀 Phase 1: Training top layers ({INITIAL_EPOCHS} epochs)")
    history1 = model.fit(
        train_data,
        epochs=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning last 50 layers (like face classifier)
    print(f"\n🔧 Phase 2: Fine-tuning last 50 layers ({FINE_TUNE_EPOCHS} epochs)")
    
    # Unfreeze last 50 layers exactly like face classifier
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=FINE_TUNE_RATE),
        loss=loss,
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_data,
        epochs=EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,  # Use same callbacks as face classifier
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
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"📈 {model_name} - Accuracy: {accuracy*100:.2f}%")

    # Ensure save directory exists before saving
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # Save Confusion Matrix (simplified for multi-class)
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"{model_name} Iris Person Recognition")
        plt.xlabel("Predicted Person")
        plt.ylabel("Actual Person")
        plt.colorbar()
        confusion_matrix_path = f"{SAVE_PATH}/{model_name}_confusion_matrix.png"
        print(f"💾 Saving confusion matrix to: {confusion_matrix_path}")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not save confusion matrix for {model_name}: {e}")
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

    return accuracy, cm

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

    print(f"\n🏁 Starting HIGH-ACCURACY iris person recognition for {len(models)} models...")
    print(f"🎯 Target Accuracy: 80-95% (Real Iris Person Recognition)")
    print(f"👥 Recognizing {num_classes} different people")
    print(f"⚙️  Configuration: {EPOCHS} epochs, batch size {BATCH_SIZE}, image size {IMG_SIZE}")
    print(f"🔧 Two-Phase Training: {INITIAL_EPOCHS} initial + {FINE_TUNE_EPOCHS} fine-tune (50 layers)")

    # Train each model with enhanced strategy
    for name, fn in models.items():
        try:
            accuracy, cm = train_and_evaluate(fn, name)
            results.append({
                "Model": name,
                "Accuracy": f"{accuracy*100:.2f}%",
                "Accuracy_Score": accuracy
            })
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            continue

    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="Accuracy_Score", ascending=False)
        
        df.to_csv(f"{SAVE_PATH}/iris_recognition_results.csv", index=False)
        
        print(f"\n🏆 Final Iris Person Recognition Results:")
        print("=" * 60)
        print(df[['Model', 'Accuracy']].to_string(index=False))
        print("=" * 60)
        
        # Show best model
        best_model = df.iloc[0]
        best_accuracy = best_model['Accuracy_Score'] * 100
        print(f"🥇 Best model: {best_model['Model']} (Accuracy: {best_model['Accuracy']})")
        
        # Check if we achieved target
        if best_accuracy >= 80:
            print(f"🎉 SUCCESS! Achieved {best_accuracy:.1f}% accuracy (target: 80%+)")
        else:
            print(f"⚠️  Below target: {best_accuracy:.1f}% accuracy (target: 80%+)")
    else:
        print("❌ No models completed successfully")

    print(f"\n✅ Iris person recognition training complete! Check {SAVE_PATH} for all results.")