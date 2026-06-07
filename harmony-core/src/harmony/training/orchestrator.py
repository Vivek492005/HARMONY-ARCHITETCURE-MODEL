import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from harmony.models.harmony_model import HarmonyModel

class TrainingOrchestrator:
    """
    Manages the multi-stage training pipeline for HARMONY-Core:
      - Stage 1: Backbone pretraining (Language Modeling loss)
      - Stage 2: Retrieval Memory setup (Indexing phase)
      - Stage 3: Planner training (Cross Entropy loss)
      - Stage 4: Verifier training (Binary Cross Entropy loss)
      - Stage 5: Joint fine-tuning (combined multi-task learning)
    Supports checkpoint saving, loading, and resuming training.
    """
    def __init__(self, model: HarmonyModel, device: str = "cpu", checkpoint_dir: str = "checkpoints"):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model.to(device)
        
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, stage_name: str, epoch: int, step: int, optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Saves a model checkpoint along with optimizer state.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"harmony_{stage_name}_epoch{epoch}.pt")
        state = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "step": step
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
            
        torch.save(state, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, filepath: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Loads a checkpoint from disk.
        """
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        return {
            "epoch": checkpoint.get("epoch", 0),
            "step": checkpoint.get("step", 0)
        }

    def train_stage_1_backbone(self, dataloader: DataLoader, epochs: int = 3, lr: float = 1e-4) -> float:
        """
        Stage 1: Backbone pretraining (Next-Token Prediction on Chunk representations).
        Trains embedding, local context mixer, selective state backbone, and generator.
        """
        print("--- Starting Stage 1: Backbone Pretraining ---")
        self.model.train()
        
        # Freeze planner and verifier heads to prevent distortion
        for p in self.model.planner.parameters():
            p.requires_grad = False
        for p in self.model.verifier.parameters():
            p.requires_grad = False
            
        # Optimize backbone parts
        backbone_params = (
            list(self.model.embedding.parameters()) +
            list(self.model.local_mixer.parameters()) +
            list(self.model.backbone.parameters()) +
            list(self.model.generator.parameters())
        )
        optimizer = torch.optim.AdamW(backbone_params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        step_idx = 0
        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device) # (batch, num_chunks, chunk_size)
                target_ids = batch["target_ids"].to(self.device) # (batch, num_chunks)
                
                optimizer.zero_grad()
                logits, _ = self.model(input_ids) # (batch, num_chunks, vocab_size)
                
                # Compute LM loss
                vocab_size = logits.size(-1)
                loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step_idx += 1
                
            avg_loss = total_loss / len(dataloader)
            print(f"Stage 1 | Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            self.save_checkpoint("stage1_backbone", epoch+1, step_idx, optimizer)
            final_loss = avg_loss
            
        return final_loss

    def train_stage_3_planner(self, dataloader: DataLoader, epochs: int = 3, lr: float = 1e-4) -> float:
        """
        Stage 3: Planner training (Supervised task planning using target action choices).
        """
        print("--- Starting Stage 3: Planner Head Training ---")
        self.model.train()
        
        # Unfreeze planner
        for p in self.model.planner.parameters():
            p.requires_grad = True
            
        optimizer = torch.optim.AdamW(self.model.planner.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        step_idx = 0
        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for states, targets in dataloader:
                states = states.to(self.device)
                targets = targets.to(self.device).long()
                
                optimizer.zero_grad()
                logits = self.model.planner(states) # (batch, num_actions)
                loss = loss_fn(logits, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step_idx += 1
                
            avg_loss = total_loss / len(dataloader)
            print(f"Stage 3 | Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            self.save_checkpoint("stage3_planner", epoch+1, step_idx, optimizer)
            final_loss = avg_loss
            
        return final_loss

    def train_stage_4_verifier(self, dataloader: DataLoader, epochs: int = 3, lr: float = 1e-4) -> float:
        """
        Stage 4: Verifier training (Calibration of confidence outputs).
        """
        print("--- Starting Stage 4: Verifier Head Training ---")
        self.model.train()
        
        # Unfreeze verifier
        for p in self.model.verifier.parameters():
            p.requires_grad = True
            
        optimizer = torch.optim.AdamW(self.model.verifier.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        
        step_idx = 0
        final_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for states, targets in dataloader:
                states = states.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)
                
                optimizer.zero_grad()
                confidence = self.model.verifier(states)
                loss = loss_fn(confidence, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step_idx += 1
                
            avg_loss = total_loss / len(dataloader)
            print(f"Stage 4 | Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
            self.save_checkpoint("stage4_verifier", epoch+1, step_idx, optimizer)
            final_loss = avg_loss
            
        return final_loss

    def train_stage_5_joint(
        self,
        lm_loader: DataLoader,
        planner_loader: DataLoader,
        verifier_loader: DataLoader,
        epochs: int = 3,
        lr: float = 1e-5,
        w_lm: float = 1.0,
        w_plan: float = 0.5,
        w_ver: float = 0.5
    ) -> Dict[str, float]:
        """
        Stage 5: Joint Fine-Tuning.
        Trains all components simultaneously, balancing LM, planning, and verification.
        """
        print("--- Starting Stage 5: Joint Fine-Tuning ---")
        self.model.train()
        
        # Enable gradients everywhere
        for p in self.model.parameters():
            p.requires_grad = True
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        lm_loss_fn = nn.CrossEntropyLoss()
        plan_loss_fn = nn.CrossEntropyLoss()
        ver_loss_fn = nn.BCELoss()
        
        step_idx = 0
        epoch_losses = {}
        
        for epoch in range(epochs):
            total_lm_loss = 0.0
            total_plan_loss = 0.0
            total_ver_loss = 0.0
            
            # Use zip to iterate over loaders simultaneously.
            # We wrap loaders in cycle to prevent short iteration.
            lm_iter = iter(lm_loader)
            plan_iter = iter(planner_loader)
            ver_iter = iter(verifier_loader)
            
            num_steps = max(len(lm_loader), len(planner_loader), len(verifier_loader))
            
            for _ in range(num_steps):
                optimizer.zero_grad()
                joint_loss = torch.tensor(0.0, device=self.device)
                
                # A. Language Modeling Batch
                try:
                    lm_batch = next(lm_iter)
                except StopIteration:
                    lm_iter = iter(lm_loader)
                    lm_batch = next(lm_iter)
                    
                input_ids = lm_batch["input_ids"].to(self.device)
                target_ids = lm_batch["target_ids"].to(self.device)
                logits, _ = self.model(input_ids)
                vocab_size = logits.size(-1)
                lm_loss = lm_loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
                joint_loss += w_lm * lm_loss
                total_lm_loss += lm_loss.item()
                
                # B. Planner Batch
                try:
                    plan_batch = next(plan_iter)
                except StopIteration:
                    plan_iter = iter(planner_loader)
                    plan_batch = next(plan_iter)
                    
                plan_states, plan_targets = plan_batch
                plan_states = plan_states.to(self.device)
                plan_targets = plan_targets.to(self.device).long()
                plan_logits = self.model.planner(plan_states)
                plan_loss = plan_loss_fn(plan_logits, plan_targets)
                joint_loss += w_plan * plan_loss
                total_plan_loss += plan_loss.item()
                
                # C. Verifier Batch
                try:
                    ver_batch = next(ver_iter)
                except StopIteration:
                    ver_iter = iter(verifier_loader)
                    ver_batch = next(ver_iter)
                    
                ver_states, ver_targets = ver_batch
                ver_states = ver_states.to(self.device)
                ver_targets = ver_targets.to(self.device).float().view(-1, 1)
                ver_confidence = self.model.verifier(ver_states)
                ver_loss = ver_loss_fn(ver_confidence, ver_targets)
                joint_loss += w_ver * ver_loss
                total_ver_loss += ver_loss.item()
                
                # Backward pass
                joint_loss.backward()
                optimizer.step()
                
                step_idx += 1
                
            avg_lm = total_lm_loss / num_steps
            avg_plan = total_plan_loss / num_steps
            avg_ver = total_ver_loss / num_steps
            print(f"Stage 5 | Epoch {epoch+1}/{epochs} | LM Loss: {avg_lm:.4f} | Plan Loss: {avg_plan:.4f} | Ver Loss: {avg_ver:.4f}")
            self.save_checkpoint("stage5_joint", epoch+1, step_idx, optimizer)
            epoch_losses = {"lm_loss": avg_lm, "plan_loss": avg_plan, "ver_loss": avg_ver}
            
        return epoch_losses
