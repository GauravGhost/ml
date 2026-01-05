#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path

from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3,
    DenseNet121, EfficientNetB0, Xception
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report

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

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 35

FACE_DATA_PATH = "./data/face"
PROCESSED_DATA_PATH = "./data/face_processed"
SAVE_PATH = "./results_face"

os.makedirs(SAVE_PATH, exist_ok=True)

def organize_face_data(classification_type="binary"):
    """
    Organize face data for training
    classification_type: "binary" (live vs spoof) or "multiclass" (different spoof types)
    """
    print(f"🔧 Organizing face data for {classification_type} classification...")
    
    # Remove existing processed directory
    if os.path.exists(PROCESSED_DATA_PATH):
        shutil.rmtree(PROCESSED_DATA_PATH)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    if classification_type == "binary":
        # Binary: Live vs Spoof
        live_path = os.path.join(PROCESSED_DATA_PATH, "live")
        spoof_path = os.path.join(PROCESSED_DATA_PATH, "spoof")
        os.makedirs(live_path, exist_ok=True)
        os.makedirs(spoof_path, exist_ok=True)
        
        # Copy live subjects
        live_source = os.path.join(FACE_DATA_PATH, "live_subject_images")
        if os.path.exists(live_source):
            for file in os.listdir(live_source):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(
                        os.path.join(live_source, file),
                        os.path.join(live_path, f"live_{file}")
                    )
        
        # Copy all spoof categories
        spoof_categories = [
            "bobblehead_images", "filament_projection_images", "Full_cloth_mask_images",
            "half_cloth_mask_images", "HQ_3D_MASK_images", "mannequin_projection_images",
            "resin_projection_images", "white_filament_images", "white_mannequin_images",
            "white_resin_images"
        ]
        
        file_count = 0
        for category in spoof_categories:
            category_path = os.path.join(FACE_DATA_PATH, category)
            if os.path.exists(category_path):
                # Handle both direct images and device-specific folders
                if any(d in os.listdir(category_path) for d in ["DSLR", "IPHONE14", "SAMSUNG_S9"]):
                    # Device-specific folders
                    for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                        device_path = os.path.join(category_path, device)
                        if os.path.exists(device_path):
                            for file in os.listdir(device_path):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    shutil.copy2(
                                        os.path.join(device_path, file),
                                        os.path.join(spoof_path, f"{category}_{device}_{file}")
                                    )
                                    file_count += 1
                else:
                    # Direct images
                    for file in os.listdir(category_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            shutil.copy2(
                                os.path.join(category_path, file),
                                os.path.join(spoof_path, f"{category}_{file}")
                            )
                            file_count += 1
        
        print(f"✅ Organized {file_count} spoof images and live images for binary classification")
        
    else:  # multiclass
        # Multi-class: Different spoof types + live
        categories = {
            "live": "live_subject_images",
            "bobblehead": "bobblehead_images",
            "filament_projection": "filament_projection_images",
            "full_cloth_mask": "Full_cloth_mask_images",
            "half_cloth_mask": "half_cloth_mask_images",
            "hq_3d_mask": "HQ_3D_MASK_images",
            "mannequin_projection": "mannequin_projection_images",
            "resin_projection": "resin_projection_images",
            "white_filament": "white_filament_images",
            "white_mannequin": "white_mannequin_images",
            "white_resin": "white_resin_images"
        }
        
        for class_name, folder_name in categories.items():
            class_path = os.path.join(PROCESSED_DATA_PATH, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            source_path = os.path.join(FACE_DATA_PATH, folder_name)
            if os.path.exists(source_path):
                file_count = 0
                # Handle both direct images and device-specific folders
                if any(d in os.listdir(source_path) for d in ["DSLR", "IPHONE14", "SAMSUNG_S9"]):
                    # Device-specific folders
                    for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                        device_path = os.path.join(source_path, device)
                        if os.path.exists(device_path):
                            for file in os.listdir(device_path):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    shutil.copy2(
                                        os.path.join(device_path, file),
                                        os.path.join(class_path, f"{device}_{file}")
                                    )
                                    file_count += 1
                else:
                    # Direct images
                    for file in os.listdir(source_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            shutil.copy2(
                                os.path.join(source_path, file),
                                os.path.join(class_path, file)
                            )
                            file_count += 1
                
                print(f"✅ {class_name}: {file_count} images")
    
    return PROCESSED_DATA_PATH

def setup_data_generators(dataset_path):
    """Setup data generators for training and validation"""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    # Get class information
    class_folders = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')]
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
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=class_mode,
        subset='training',
        shuffle=True
    )
    
    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=class_mode,
        subset='validation',
        shuffle=False
    )
    
    print(f"🎯 Found {train_data.samples} training samples")
    print(f"🎯 Found {val_data.samples} validation samples")
    print(f"📊 Classes: {train_data.class_indices}")
    
    return train_data, val_data, num_classes, class_mode, activation, loss, output_units

def train_and_evaluate(model_fn, model_name, train_data, val_data, num_classes, class_mode, activation, loss, output_units):
    """Train and evaluate a model"""
    print(f"\n🚀 Training {model_name}")
    
    base_model = model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = Dense(output_units, activation=activation)(x)

    model = Model(base_model.input, output)

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )

    print(f"📊 {model_name} - Total parameters: {model.count_params():,}")

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        steps_per_epoch=min(100, train_data.samples // BATCH_SIZE),
        validation_steps=min(30, val_data.samples // BATCH_SIZE),
        verbose=1
    )

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
        
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        # Print classification report
        print(f"\n📊 {model_name} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Live', 'Spoof']))
        
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
        
        # Print classification report
        class_names = list(train_data.class_indices.keys())
        print(f"\n📊 {model_name} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"📈 {model_name} - AUC: {auc:.4f}")

    # Save model
    model.save(f"{SAVE_PATH}/{model_name}.h5")
    print(f"💾 Model saved as {model_name}.h5")

    # Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.savefig(f"{SAVE_PATH}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve for binary classification
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

    return {
        'model_name': model_name,
        'auc': auc,
        'accuracy': history.history['val_accuracy'][-1],
        'loss': history.history['val_loss'][-1],
        'confusion_matrix': cm.tolist()
    }

def main():
    """Main function to run face classification training"""
    print("🎭 Face Anti-Spoofing Classification Training")
    print("=" * 50)
    
    # Ask user for classification type
    print("\nChoose classification type:")
    print("1. Binary (Live vs Spoof)")
    print("2. Multi-class (Different spoof types)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        classification_type = "binary"
    elif choice == "2":
        classification_type = "multiclass"
    else:
        print("Invalid choice. Using binary classification by default.")
        classification_type = "binary"
    
    # Organize data
    dataset_path = organize_face_data(classification_type)
    
    # Setup data generators
    train_data, val_data, num_classes, class_mode, activation, loss, output_units = setup_data_generators(dataset_path)
    
    # Models to train
    models_to_train = [
        (ResNet50, "ResNet50"),
        (VGG16, "VGG16"),
        (InceptionV3, "InceptionV3"),
        (DenseNet121, "DenseNet121"),
        (EfficientNetB0, "EfficientNetB0"),
        (Xception, "Xception")
    ]
    
    results = []
    
    # Train each model
    for model_fn, model_name in models_to_train:
        try:
            result = train_and_evaluate(model_fn, model_name, train_data, val_data, num_classes, 
                                      class_mode, activation, loss, output_units)
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Save comparison results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{SAVE_PATH}/model_comparison_results.csv", index=False)
        
        print("\n🏆 Model Comparison Summary:")
        print("=" * 50)
        for _, row in results_df.iterrows():
            print(f"{row['model_name']}: AUC={row['auc']:.4f}, Accuracy={row['accuracy']:.4f}")
        
        best_model = results_df.loc[results_df['auc'].idxmax()]
        print(f"\n🥇 Best Model: {best_model['model_name']} (AUC: {best_model['auc']:.4f})")
    
    print(f"\n✅ Training completed! Results saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()