#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3,
    DenseNet121, EfficientNetB0, Xception
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Enable GPU if available
print("🔧 Setting up GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU enabled: {len(gpus)} GPU(s) found")
    except RuntimeError as e:
        print(f"❌ GPU setup error: {e}")
else:
    print("⚠️  No GPU found, using CPU")

# Configuration - Update these paths for your local setup
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

# Update these paths to your local dataset location
DATASET_PATH = "./data/fingerprint"  # Local dataset path
SAVE_PATH = "./results"

os.makedirs(SAVE_PATH, exist_ok=True)

print(f"📁 Dataset path: {DATASET_PATH}")
print(f"💾 Results will be saved to: {SAVE_PATH}")

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found at {DATASET_PATH}")
    print("📋 Please update DATASET_PATH variable to point to your fingerprint dataset")
    print("   Expected structure: DATASET_PATH/class1/, DATASET_PATH/class2/, etc.")
    exit(1)

# Detect number of classes
class_folders = [d for d in os.listdir(DATASET_PATH) 
                if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith('.')]
num_classes = len(class_folders)

print(f"📂 Found {num_classes} classes: {class_folders}")

# Determine class mode
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

val_data = datagen.flow_from_directory(
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
    """Train and evaluate a model - exact copy of original Colab function"""
    print(f"\n🚀 Training {model_name}")
    
    # Create base model
    base_model = model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3)
    )

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(output_units, activation=activation)(x)

    model = Model(base_model.input, output)

    # Compile model
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")

    # Train model
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        steps_per_epoch=100,
        validation_steps=30,
        verbose=1
    )

    # Make predictions
    val_data.reset()
    y_true = val_data.classes
    y_pred_proba = model.predict(val_data, verbose=0)
    
    if class_mode == 'binary':
        # Binary classification
        y_prob = y_pred_proba.ravel()
        y_pred = (y_prob > 0.5).astype(int)
        
        # Ensure arrays are same length
        min_len = min(len(y_true), len(y_prob))
        y_true = y_true[:min_len]
        y_prob = y_prob[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
    else:
        # Multi-class classification
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Ensure arrays are same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_proba = y_pred_proba[:min_len]
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # For multi-class AUC, use one-vs-rest
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
        except:
            auc = 0.0  # If AUC calculation fails
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
        # Multi-class ROC curve
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(num_classes):
            if i < len(colors):
                color = colors[i]
            else:
                color = np.random.rand(3,)
            
            # Binary classification for each class
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
    # Model definitions - same as original
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

    # Train each model - same as original
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

    # Save results - same as original
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