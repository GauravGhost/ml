"""
Utilities Package

Common utilities used across different classifiers
"""

from .analyze_results import analyze_classifier_results, analyze_all_classifiers

# Convenience functions
def analyze_fingerprint():
    """Analyze fingerprint classifier results"""
    return analyze_classifier_results("fingerprint")

def analyze_face():
    """Analyze face classifier results"""
    return analyze_classifier_results("face")

def analyze_iris():
    """Analyze iris classifier results"""
    return analyze_classifier_results("iris")