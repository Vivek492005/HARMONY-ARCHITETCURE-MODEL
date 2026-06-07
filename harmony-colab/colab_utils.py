"""
Utility functions for Google Colab - HARMONY Model
Handles Google Drive integration, checkpointing, and monitoring
"""

import os
import torch
import shutil
from pathlib import Path
from datetime import datetime
from google.colab import drive
import json

class ColabManager:
    """Manages Colab-specific operations for HARMONY training"""
    
    def __init__(self, drive_path="/content/drive/MyDrive/HARMONY"):
        self.drive_path = drive_path
        self.checkpoint_dir = "/content/checkpoints"
        self.mounted = False
        
    def mount_drive(self):
        """Mount Google Drive for model persistence"""
        try:
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
            self.mounted = True
            
            # Create HARMONY directory if it doesn't exist
            os.makedirs(self.drive_path, exist_ok=True)
            print(f"✓ Google Drive mounted at {self.drive_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to mount Google Drive: {e}")
            print("Continuing without Drive (models will be lost after session)")
            return False
    
    def save_checkpoint(self, model, stage_name, optimizer=None, epoch=None):
        """Save model checkpoint to both local and Drive"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{stage_name}_{timestamp}.pt"
        local_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'stage': stage_name,
            'timestamp': timestamp,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        torch.save(checkpoint, local_path)
        print(f"✓ Checkpoint saved locally: {local_path}")
        
        # Copy to Drive if mounted
        if self.mounted:
            drive_path = os.path.join(self.drive_path, checkpoint_name)
            shutil.copy2(local_path, drive_path)
            print(f"✓ Checkpoint saved to Drive: {drive_path}")
        
        return local_path
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
            print(f"  Stage: {checkpoint.get('stage', 'unknown')}")
            print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
            return checkpoint
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self):
        """List available checkpoints"""
        checkpoints = []
        
        # Check local checkpoints
        if os.path.exists(self.checkpoint_dir):
            local_files = list(Path(self.checkpoint_dir).glob("*.pt"))
            checkpoints.extend([("local", str(f)) for f in local_files])
        
        # Check Drive checkpoints
        if self.mounted and os.path.exists(self.drive_path):
            drive_files = list(Path(self.drive_path).glob("*.pt"))
            checkpoints.extend([("drive", str(f)) for f in drive_files])
        
        return checkpoints
    
    def get_gpu_info(self):
        """Get GPU information"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                'available': True,
                'name': device_name,
                'memory_gb': device_memory,
                'device': 'cuda'
            }
        else:
            return {
                'available': False,
                'name': 'CPU',
                'memory_gb': 0,
                'device': 'cpu'
            }
    
    def monitor_memory(self):
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
            }
        return None
    
    def print_memory_status(self):
        """Print current memory status"""
        gpu_info = self.get_gpu_info()
        print("\n" + "=" * 60)
        print("GPU/Memory Status")
        print("=" * 60)
        print(f"Device: {gpu_info['name']}")
        if gpu_info['available']:
            print(f"Total Memory: {gpu_info['memory_gb']:.2f} GB")
            
            memory = self.monitor_memory()
            if memory:
                print(f"Allocated: {memory['allocated_gb']:.2f} GB")
                print(f"Reserved: {memory['reserved_gb']:.2f} GB")
                print(f"Free: {memory['free_gb']:.2f} GB")
        print("=" * 60 + "\n")
    
    def save_training_log(self, log_data, log_name="training_log.json"):
        """Save training log to Drive"""
        if self.mounted:
            log_path = os.path.join(self.drive_path, log_name)
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"✓ Training log saved to {log_path}")
    
    def download_from_drive(self, filename, destination="/content"):
        """Download a file from Google Drive"""
        if self.mounted:
            source = os.path.join(self.drive_path, filename)
            if os.path.exists(source):
                shutil.copy2(source, destination)
                print(f"✓ Downloaded {filename} from Drive")
                return True
            else:
                print(f"✗ File not found in Drive: {filename}")
                return False
        return False

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")

def print_progress(current, total, prefix="Progress"):
    """Print progress bar"""
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix}: [{bar}] {percent:.1f}% ({current}/{total})", end='')
    if current == total:
        print()  # New line when complete

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'sentence_transformers': 'Sentence Transformers',
        'faiss': 'FAISS',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Please run setup_colab.py first")
        return False
    
    print("\n✓ All dependencies installed!")
    return True
