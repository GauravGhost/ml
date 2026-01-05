#!/usr/bin/env python3
"""
Iris Model Usage Script

This script demonstrates how to use the trained iris classifier model.
"""

import os
import sys
import numpy as np
from iris_classifier import IrisClassifier

def main():
    """
    Main function to use the iris classifier
    """
    # Initialize classifier
    classifier = IrisClassifier()
    
    # Path to saved model
    model_path = "../../results/iris/iris_model.joblib"
    
    if os.path.exists(model_path):
        # Load pre-trained model
        classifier.load_model(model_path)
        
        # TODO: Add logic to load test data and make predictions
        print("Iris model loaded successfully!")
        print("Ready to make predictions on new iris data.")
        
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first using iris_classifier.py")

if __name__ == "__main__":
    main()