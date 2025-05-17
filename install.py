#!/usr/bin/env python3
"""
Installation script for ATS Resume Checker that handles dependency issues
"""

import subprocess
import sys
import os
import platform

def install_dependencies():
    """Install dependencies one by one with special handling for problematic packages."""
    print("Installing dependencies...")
    
    # Install numpy first (required by scikit-learn)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
    # Install special handling for PyMuPDF based on platform
    pymupdf_installed = False
    system = platform.system()
    
    print(f"Detected platform: {system}")
    
    if system == "Darwin":  # macOS
        try:
            # Use an older version that's more compatible with macOS
            print("Trying to install PyMuPDF 1.19.0 (more compatible with macOS)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--use-pep517", "pymupdf==1.19.0"])
            pymupdf_installed = True
        except subprocess.CalledProcessError:
            print("Failed to install PyMuPDF 1.19.0")
            
        if not pymupdf_installed:
            try:
                # Try with homebrew support if available
                print("Attempting to install with homebrew dependencies...")
                subprocess.check_call(["brew", "install", "mupdf"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--use-pep517", "pymupdf==1.19.0"])
                pymupdf_installed = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Homebrew installation attempt failed")
    
    # Try the alternative PDF processing library for all platforms if PyMuPDF fails
    if not pymupdf_installed:
        try:
            print("Trying alternative PDF library (pdfminer.six)...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfminer.six"])
            # Create a marker file to indicate we're using the alternative
            with open("using_pdfminer.txt", "w") as f:
                f.write("This installation is using pdfminer.six instead of PyMuPDF")
            print("Successfully installed pdfminer.six as an alternative to PyMuPDF")
        except subprocess.CalledProcessError:
            print("Warning: Could not install any PDF processing library. PDF files will not be supported.")
    
    # Install the rest of the requirements
    packages = [
        "flask==2.3.3",
        "python-docx==0.8.11",
        "scikit-learn==1.3.0",
        "nltk==3.8.1", 
        "python-dotenv==1.0.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Warning: Could not install {package}")
    
    # Install spaCy separately
    try:
        print("Installing spaCy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.7.2"])
        # Download spaCy model
        print("Downloading spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        print("Successfully installed spaCy and the required model")
    except subprocess.CalledProcessError:
        print("Warning: Could not install spaCy or its model. You may need to install it manually.")

def setup_directories():
    """Create necessary directories."""
    os.makedirs('uploads', exist_ok=True)
    print("Upload directory created.")

def main():
    """Main installation function."""
    print("Setting up ATS Resume Checker...")
    
    install_dependencies()
    setup_directories()
    
    print("\nSetup complete! You can now run the app with: python app.py")
    print("If you encountered any errors, please try installing the problematic packages manually.")

if __name__ == "__main__":
    main() 