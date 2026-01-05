#!/usr/bin/env python3
"""
Standalone Analysis Script

Quick analysis script that can be run directly to analyze classifier results.
This is a convenience wrapper around the main analysis functionality.
"""

import sys
import os
from utils.analyze_results import main

if __name__ == "__main__":
    main()