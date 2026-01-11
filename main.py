#!/usr/bin/env python3
"""
Main Project Runner

This script provides a unified interface to run different biometric classifiers.
"""

import os
import sys
import argparse

def run_fingerprint_classifier():
    """Run fingerprint classifier"""
    print("Running Fingerprint Classifier...")
    os.chdir("classifiers/fingerprint")
    os.system("python fingerprint_classifier.py")
    os.chdir("../..")

def run_face_classifier():
    """Run face classifier"""
    print("Running Face Classifier...")
    os.chdir("classifiers/face")
    os.system("python face_classifier.py")
    os.chdir("../..")

def run_iris_classifier():
    """Run iris classifier"""
    print("Running Iris Classifier...")
    os.chdir("classifiers/iris")
    os.system("python iris_classifier.py")
    os.chdir("../..")

def run_unified_classifier():
    """Run unified multi-modal classifier"""
    print("Running Unified Multi-Modal Classifier...")
    os.chdir("classifiers")
    os.system("python unified_multimodal_classifier.py")
    os.chdir("..")

def run_unified_classifier_with_model(model_name):
    """Run unified multi-modal classifier with specific model"""
    print(f"Running Unified Multi-Modal Classifier with {model_name}...")
    os.chdir("classifiers")
    os.system(f"python unified_multimodal_classifier.py --model {model_name}")
    os.chdir("..")

def use_fingerprint_model():
    """Use trained fingerprint model"""
    print("Using Fingerprint Model...")
    os.chdir("classifiers/fingerprint")
    os.system("python use_model.py")
    os.chdir("../..")

def use_face_model():
    """Use trained face model"""
    print("Using Face Model...")
    os.chdir("classifiers/face")
    os.system("python use_face_model.py")
    os.chdir("../..")

def use_iris_model():
    """Use trained iris model"""
    print("Using Iris Model...")
    os.chdir("classifiers/iris")
    os.system("python use_iris_model.py")
    os.chdir("../..")

def use_unified_model():
    """Use trained unified model"""
    print("Using Unified Multi-Modal Model...")
    os.chdir("classifiers")
    os.system("python use_unified_model.py")
    os.chdir("..")

def analyze_fingerprint_results():
    """Analyze fingerprint classifier results"""
    print("Analyzing Fingerprint Results...")
    os.system("python3 utils/analyze_results.py -c fingerprint")

def analyze_face_results():
    """Analyze face classifier results"""
    print("Analyzing Face Results...")
    os.system("python3 utils/analyze_results.py -c face")

def analyze_iris_results():
    """Analyze iris classifier results"""
    print("Analyzing Iris Results...")
    os.system("python3 utils/analyze_results.py -c iris")

def analyze_unified_results():
    """Analyze unified classifier results"""
    print("Analyzing Unified Results...")
    os.system("python3 utils/analyze_results.py -c unified")

def analyze_all_results():
    """Analyze all classifier results"""
    print("Analyzing All Results...")
    os.system("python3 utils/analyze_results.py -c all")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Biometric Classifiers Project Runner")
    parser.add_argument("--classifier", "-c", choices=["fingerprint", "face", "iris", "unified"], 
                       help="Choose classifier to run")
    parser.add_argument("--action", "-a", choices=["train", "use", "analyze"], 
                       default="train", help="Choose action: train, use, or analyze")
    parser.add_argument("--model", "-m", 
                       choices=["ResNet50", "VGG16", "InceptionV3", "DenseNet121", "EfficientNetB0", "Xception"],
                       help="Specific model to train (only for unified classifier)")
    
    args = parser.parse_args()
    
    if not args.classifier and args.action != "analyze":
        print("Available classifiers:")
        print("1. Fingerprint Classifier")
        print("2. Face Classifier") 
        print("3. Iris Classifier")
        print("4. Unified Multi-Modal Classifier (ALL DATASETS)")
        print("\nUsage examples:")
        print("python main.py -c fingerprint -a train")
        print("python main.py -c face -a use")
        print("python main.py -c iris -a train")
        print("python main.py -c unified -a train          # Train ALL 6 models")
        print("python main.py -c unified -a train -m EfficientNetB0  # Train specific model")
        print("python main.py -c unified -a use            # Use best unified model")
        print("python main.py -c fingerprint -a analyze    # Analyze specific")
        print("python main.py -a analyze                   # Analyze all")
        return
    
    if args.action == "train":
        if args.classifier == "fingerprint":
            run_fingerprint_classifier()
        elif args.classifier == "face":
            run_face_classifier()
        elif args.classifier == "iris":
            run_iris_classifier()
        elif args.classifier == "unified":
            if args.model:
                run_unified_classifier_with_model(args.model)
            else:
                run_unified_classifier()
    
    elif args.action == "use":
        if args.classifier == "fingerprint":
            use_fingerprint_model()
        elif args.classifier == "face":
            use_face_model()
        elif args.classifier == "iris":
            use_iris_model()
        elif args.classifier == "unified":
            use_unified_model()
    
    elif args.action == "analyze":
        if args.classifier == "fingerprint":
            analyze_fingerprint_results()
        elif args.classifier == "face":
            analyze_face_results()
        elif args.classifier == "iris":
            analyze_iris_results()
        elif args.classifier == "unified":
            analyze_unified_results()
        else:
            analyze_all_results()

if __name__ == "__main__":
    main()