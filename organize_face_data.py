#!/usr/bin/env python3
"""
Face Data Organization Script
Organizes face data into proper class directories for the standardized classifier
"""

import os
import shutil
from pathlib import Path

def organize_face_data(classification_type="binary"):
    """
    Organize face data for training
    classification_type: "binary" (live vs spoof) or "multiclass" (different spoof types)
    """
    
    FACE_DATA_PATH = "./data/face"
    PROCESSED_DATA_PATH = "./data/face_organized"
    
    print(f"🔧 Organizing face data for {classification_type} classification...")
    print(f"📁 Source: {FACE_DATA_PATH}")
    print(f"📁 Target: {PROCESSED_DATA_PATH}")
    
    # Remove existing processed directory
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"🗑️  Removing existing organized data at {PROCESSED_DATA_PATH}")
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
        live_count = 0
        if os.path.exists(live_source):
            for file in os.listdir(live_source):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    shutil.copy2(
                        os.path.join(live_source, file),
                        os.path.join(live_path, f"live_{file}")
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
        for category in spoof_categories:
            category_path = os.path.join(FACE_DATA_PATH, category)
            if os.path.exists(category_path):
                print(f"   📂 Processing {category}...")
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
        
        print(f"✅ Binary classification organized:")
        print(f"   📊 Live samples: {live_count}")
        print(f"   📊 Spoof samples: {spoof_count}")
        print(f"   📊 Total samples: {live_count + spoof_count}")
        
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
        
        total_samples = 0
        for class_name, folder_name in categories.items():
            class_path = os.path.join(PROCESSED_DATA_PATH, class_name)
            os.makedirs(class_path, exist_ok=True)
            
            source_path = os.path.join(FACE_DATA_PATH, folder_name)
            if os.path.exists(source_path):
                file_count = 0
                print(f"   📂 Processing {class_name} from {folder_name}...")
                # Handle both direct images and device-specific folders
                if any(d in os.listdir(source_path) for d in ["DSLR", "IPHONE14", "SAMSUNG_S9"]):
                    # Device-specific folders
                    for device in ["DSLR", "IPHONE14", "SAMSUNG_S9"]:
                        device_path = os.path.join(source_path, device)
                        if os.path.exists(device_path):
                            for file in os.listdir(device_path):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    shutil.copy2(
                                        os.path.join(device_path, file),
                                        os.path.join(class_path, f"{device}_{file}")
                                    )
                                    file_count += 1
                else:
                    # Direct images
                    for file in os.listdir(source_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            shutil.copy2(
                                os.path.join(source_path, file),
                                os.path.join(class_path, file)
                            )
                            file_count += 1
                
                print(f"      📊 {class_name}: {file_count} images")
                total_samples += file_count
            else:
                print(f"      ⚠️  {folder_name} not found, skipping...")
        
        print(f"✅ Multi-class classification organized:")
        print(f"   📊 Total classes: {len([c for c in categories.keys() if os.path.exists(os.path.join(PROCESSED_DATA_PATH, c))])}")
        print(f"   📊 Total samples: {total_samples}")
    
    # Update the face_classifier.py to use the organized data
    print(f"\n💡 To use the organized data, update DATASET_PATH in face_classifier.py to: '{PROCESSED_DATA_PATH}'")
    print(f"💡 Results will be saved to: './results/face'")
    return PROCESSED_DATA_PATH

def main():
    print("🎭 FACE DATA ORGANIZATION SCRIPT")
    print("=" * 40)
    
    # Check if face data exists
    if not os.path.exists("./data/face"):
        print("❌ Face data directory not found at ./data/face")
        print("📋 Please ensure your face data is placed in the correct directory")
        return
    
    print("\nChoose organization type:")
    print("1. Binary classification (Live vs Spoof)")
    print("2. Multi-class classification (Different spoof types)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        classification_type = "binary"
    elif choice == "2":
        classification_type = "multiclass"
    else:
        print("Invalid choice. Using binary classification by default.")
        classification_type = "binary"
    
    organized_path = organize_face_data(classification_type)
    
    print(f"\n✅ Face data organization complete!")
    print(f"📁 Organized data available at: {organized_path}")
    print(f"📝 Next steps:")
    print(f"   1. Update DATASET_PATH in face_classifier.py to: '{organized_path}'")
    print(f"   2. Run the face classifier training script")
    print(f"   3. Results will be saved to: './results/face'")

if __name__ == "__main__":
    main()