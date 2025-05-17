#!/usr/bin/env python3
"""
Setup script for ATS Resume Checker
Downloads required NLTK data and spaCy models
"""

import subprocess
import sys
import nltk
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import spacy
        import docx
        import fitz
        import flask
        import sklearn
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies with: pip install -r requirements.txt")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully.")

def download_spacy_model():
    """Download required spaCy model."""
    print("Downloading spaCy model...")
    try:
        # Using the medium English model compatible with spaCy 3.7.2
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        print("spaCy model downloaded successfully.")
    except subprocess.CalledProcessError:
        print("Error downloading spaCy model. Please run manually: python -m spacy download en_core_web_md")

def setup_directories():
    """Create necessary directories."""
    os.makedirs('uploads', exist_ok=True)
    print("Upload directory created.")

def main():
    """Main setup function."""
    print("Setting up ATS Resume Checker...")
    
    if not check_dependencies():
        return
    
    download_nltk_data()
    download_spacy_model()
    setup_directories()
    
    print("\nSetup complete! You can now run the app with: python app.py")

if __name__ == "__main__":
    main() 