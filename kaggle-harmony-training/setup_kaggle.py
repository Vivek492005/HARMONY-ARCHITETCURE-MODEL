import subprocess
import sys
import os

def install_dependencies():
    """Install required packages for Kaggle environment"""
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.3",
        "faiss-cpu>=1.7.4",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "numpy>=1.24.0",
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.2",
        "pypdf>=3.9.0",
        "python-docx>=0.8.11",
        "tqdm>=4.65.0",
        "networkx>=3.0",
        "accelerate>=0.20.0",  # For GPU acceleration
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies()
