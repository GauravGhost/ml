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

def analyze_fingerprint_results():
    """Analyze fingerprint classifier results"""
    print("Analyzing Fingerprint Results...")
    os.system("python utils/analyze_results.py -c fingerprint")

def analyze_face_results():
    """Analyze face classifier results"""
    print("Analyzing Face Results...")
    os.system("python utils/analyze_results.py -c face")

def analyze_iris_results():
    """Analyze iris classifier results"""
    print("Analyzing Iris Results...")
    os.system("python utils/analyze_results.py -c iris")

def analyze_all_results():
    """Analyze all classifier results"""
    print("Analyzing All Results...")
    os.system("python utils/analyze_results.py -c all")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Biometric Classifiers Project Runner")
    parser.add_argument("--classifier", "-c", choices=["fingerprint", "face", "iris"], 
                       help="Choose classifier to run")
    parser.add_argument("--action", "-a", choices=["train", "use", "analyze"], 
                       default="train", help="Choose action: train, use, or analyze")
    
    args = parser.parse_args()
    
    if not args.classifier and args.action != "analyze":
        print("Available classifiers:")
        print("1. Fingerprint Classifier")
        print("2. Face Classifier") 
        print("3. Iris Classifier")
        print("\nUsage examples:")
        print("python main.py -c fingerprint -a train")
        print("python main.py -c face -a use")
        print("python main.py -c iris -a train")
        print("python main.py -c fingerprint -a analyze     # Analyze specific")
        print("python main.py -a analyze                    # Analyze all")
        return
    
    if args.action == "train":
        if args.classifier == "fingerprint":
            run_fingerprint_classifier()
        elif args.classifier == "face":
            run_face_classifier()
        elif args.classifier == "iris":
            run_iris_classifier()
    
    elif args.action == "use":
        if args.classifier == "fingerprint":
            use_fingerprint_model()
        elif args.classifier == "face":
            use_face_model()
        elif args.classifier == "iris":
            use_iris_model()
    
    elif args.action == "analyze":
        if args.classifier == "fingerprint":
            analyze_fingerprint_results()
        elif args.classifier == "face":
            analyze_face_results()
        elif args.classifier == "iris":
            analyze_iris_results()
        else:
            analyze_all_results()

if __name__ == "__main__":
    main()