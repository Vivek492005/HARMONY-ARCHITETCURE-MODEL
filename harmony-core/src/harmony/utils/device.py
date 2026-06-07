import torch
import contextlib
from typing import Any, Union, Dict, List

class DeviceManager:
    """
    Handles device detection (CUDA vs CPU), transferring models/tensors,
    and managing PyTorch mixed-precision (AMP) execution contexts.
    """
    def __init__(self, use_amp: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"
        
        # Initialize GradScaler for mixed precision training
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        print(f"DeviceManager initialized. Using device: {self.device}. Mixed Precision (AMP) enabled: {self.use_amp}")

    def to_device(self, obj: Any) -> Any:
        """
        Recursively moves Tensors, dicts, lists, or PyTorch Modules to the target device.
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_device(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.to_device(v) for v in obj)
        return obj

    def amp_context(self):
        """
        Returns a context manager for automatic mixed precision (AMP) if enabled.
        """
        if self.use_amp:
            # Modern PyTorch API uses torch.amp.autocast
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="cuda")
            return torch.cuda.amp.autocast()
        return contextlib.nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scales loss if using GradScaler.
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Takes optimizer step, unscaling gradients if using GradScaler.
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
