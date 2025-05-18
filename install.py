#!/usr/bin/env python3
"""
Installation script for ATS Resume Checker that handles dependency issues
"""

import subprocess
import sys
import os
import platform
import shutil
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Install ATS Resume Checker dependencies')
    parser.add_argument('--force', action='store_true', help='Force reinstallation of all packages')
    parser.add_argument('--skip-nltk', action='store_true', help='Skip NLTK data download')
    parser.add_argument('--skip-spacy', action='store_true', help='Skip spaCy model download')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    return parser.parse_args()

def check_python_version():
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"Python 3.8 or higher is required. You have Python {major}.{minor}.")
        return False
    return True

def create_virtual_env():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            print("Virtual environment created successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to create virtual environment.")
            return False
    else:
        print("Virtual environment already exists.")
        return True

def get_pip_path():
    """Get the path to pip executable in the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join('venv', 'Scripts', 'pip')
    else:  # Unix/MacOS
        return os.path.join('venv', 'bin', 'pip')

def get_python_path():
    """Get the path to python executable in the virtual environment."""
    if os.name == 'nt':  # Windows
        return os.path.join('venv', 'Scripts', 'python')
    else:  # Unix/MacOS
        return os.path.join('venv', 'bin', 'python')

def install_dependencies(args):
    """Install dependencies one by one with special handling for problematic packages."""
    pip_cmd = get_pip_path()
    python_cmd = get_python_path()
    
    print("Installing dependencies...")
    
    # Upgrade pip first
    try:
        subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("Failed to upgrade pip. Continuing with installation...")
    
    # Install numpy first (required by scikit-learn)
    try:
        subprocess.check_call([pip_cmd, "install", "numpy"])
    except subprocess.CalledProcessError:
        print("Failed to install numpy. Some features may not work properly.")
    
    # Install special handling for PyMuPDF based on platform
    pymupdf_installed = False
    system = platform.system()
    
    print(f"Detected platform: {system}")
    
    if system == "Darwin":  # macOS
        try:
            # Use an older version that's more compatible with macOS
            print("Trying to install PyMuPDF 1.19.0 (more compatible with macOS)...")
            subprocess.check_call([pip_cmd, "install", "--use-pep517", "pymupdf==1.19.0"])
            pymupdf_installed = True
        except subprocess.CalledProcessError:
            print("Failed to install PyMuPDF 1.19.0")
            
        if not pymupdf_installed:
            try:
                # Try with homebrew support if available
                print("Attempting to install with homebrew dependencies...")
                subprocess.call(["brew", "install", "mupdf"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                subprocess.check_call([pip_cmd, "install", "--use-pep517", "pymupdf==1.19.0"])
                pymupdf_installed = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Homebrew installation attempt failed")
    else:
        # For other platforms, try the latest version
        try:
            print("Installing PyMuPDF...")
            subprocess.check_call([pip_cmd, "install", "pymupdf"])
            pymupdf_installed = True
        except subprocess.CalledProcessError:
            print("Failed to install PyMuPDF")
    
    # Try the alternative PDF processing libraries
    if not pymupdf_installed:
        try:
            print("Installing pdfminer.six...")
            subprocess.check_call([pip_cmd, "install", "pdfminer.six"])
            with open("using_pdfminer.txt", "w") as f:
                f.write("This installation is using pdfminer.six instead of PyMuPDF")
            print("Successfully installed pdfminer.six as an alternative to PyMuPDF")
        except subprocess.CalledProcessError:
            print("Warning: Could not install pdfminer.six")
    
    # Try textract as well for comprehensive document support
    try:
        print("Installing textract for additional document support...")
        subprocess.check_call([pip_cmd, "install", "textract"])
        print("Successfully installed textract")
    except subprocess.CalledProcessError:
        print("Warning: Could not install textract. Some document formats may not be supported.")
    
    # Install base requirements from file
    print("Installing main requirements...")
    try:
        subprocess.check_call([pip_cmd, "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Could not install all requirements at once. Trying individually...")
        
        # Read requirements file and install packages one by one
        with open("requirements.txt", "r") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([pip_cmd, "install", package])
            except subprocess.CalledProcessError:
                print(f"Warning: Could not install {package}")
    
    # Install dev dependencies if requested
    if args.dev:
        print("Installing development dependencies...")
        dev_packages = [
            "pytest",
            "pytest-cov",
            "flake8",
            "black",
            "sphinx",
            "sphinx-rtd-theme"
        ]
        for package in dev_packages:
            try:
                subprocess.check_call([pip_cmd, "install", package])
            except subprocess.CalledProcessError:
                print(f"Warning: Could not install {package}")
    
    # Install spaCy separately if not skipped
    if not args.skip_spacy:
        try:
            print("Installing spaCy...")
            subprocess.check_call([pip_cmd, "install", "spacy==3.7.2"])
            # Download spaCy model with progress
            print("Downloading spaCy model...")
            subprocess.check_call([python_cmd, "-m", "spacy", "download", "en_core_web_md"])
            print("Successfully installed spaCy and the required model")
        except subprocess.CalledProcessError:
            print("Warning: Could not install spaCy or its model. You may need to install it manually.")
    
    # Download NLTK data if not skipped
    if not args.skip_nltk:
        try:
            print("Downloading NLTK data...")
            # Create a Python script to download NLTK data
            with open("download_nltk_data.py", "w") as f:
                f.write("""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
print("NLTK data downloaded successfully.")
""")
            # Run the script using the virtual environment Python
            subprocess.check_call([python_cmd, "download_nltk_data.py"])
            # Remove the temporary script
            os.remove("download_nltk_data.py")
        except (subprocess.CalledProcessError, OSError):
            print("Warning: Could not download NLTK data. NLP features may not work properly.")

def setup_directories():
    """Create necessary directories."""
    directories = ['uploads', 'cache', 'logs', 'temp_storage', 'tests']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main installation function."""
    args = parse_arguments()
    
    print("Setting up ATS Resume Checker...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_env():
        sys.exit(1)
    
    setup_directories()
    install_dependencies(args)
    
    print("\nSetup complete! You can now run the app with:")
    print(f"{get_python_path()} app.py")
    print("\nIf you encountered any errors, please try the following:")
    print("1. Install the problematic packages manually")
    print("2. Run with --skip-nltk or --skip-spacy if those are causing issues")
    print("3. Check the logs directory for detailed error information")

if __name__ == "__main__":
    main() 