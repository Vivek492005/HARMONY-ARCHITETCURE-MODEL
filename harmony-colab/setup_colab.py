"""
Setup script for Google Colab - HARMONY Model
Installs all required dependencies for training and testing
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    print("=" * 60)
    print("HARMONY Model - Colab Setup")
    print("=" * 60)
    print()
    
    # Core ML dependencies
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.3",
        "accelerate>=0.20.0",
    ]
    
    # Additional dependencies
    additional_packages = [
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "datasets>=2.12.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "networkx>=3.0",
        "pypdf>=3.9.0",
        "python-docx>=0.8.11",
    ]
    
    print("Step 1: Installing core ML dependencies...")
    print("-" * 60)
    for package in packages:
        install_package(package)
    
    print()
    print("Step 2: Installing additional dependencies...")
    print("-" * 60)
    for package in additional_packages:
        install_package(package)
    
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run the colab_utils.py to verify installation")
    print("2. Start with the 'Quick Testing' section in the notebook")
    print("3. Or proceed to 'Full Training' if you have time")
    print()

if __name__ == "__main__":
    main()
