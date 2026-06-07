import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from datasets import load_dataset as hf_load_dataset

from harmony.models.harmony_model import HarmonyModel

class VerifierDataset(Dataset):
    """
    VerifierDataset maps question-generation states to verifier labels:
      - 1.0: Generation is correct / logically consistent.
      - 0.0: Generation is incorrect / contradictory.
    """
    def __init__(self, texts: List[str], labels: List[float], model: HarmonyModel):
        self.texts = texts
        self.labels = labels
        self.model = model
        self.model.eval()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        with torch.no_grad():
            res = self.model.process_text(text)
            # final_state shape: (num_layers, batch=1, hidden_size)
            # Take top layer state, squeeze batch -> (hidden_size,)
            state = res["final_state"][-1].squeeze(0)
            
        return state, label


class VerifierTrainer:
    """
    Trains the VerifierHead of the HarmonyModel using binary cross entropy loss.
    """
    def __init__(self, model: HarmonyModel, lr: float = 1e-4, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.verifier.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.BCELoss()
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for states, targets in dataloader:
            states = states.to(self.device)
            # BCE loss expects float target tensor of shape (batch, 1)
            targets = targets.float().to(self.device).view(-1, 1)
            
            self.optimizer.zero_grad()
            confidence = self.model.verifier(states) # (batch, 1)
            loss = self.loss_fn(confidence, targets)
            
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.verifier.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        return avg_loss

    @staticmethod
    def prepare_dummy_data() -> Tuple[List[str], List[float]]:
        """
        Fallback data for bootstrapping verifier training.
        """
        texts = [
            "Question: What is 2+2? Answer: 4",      # Correct
            "Question: What is 2+2? Answer: 5",      # Incorrect
            "Question: Paris is capital of France",  # Correct
            "Question: Paris is capital of Spain",   # Incorrect
            "Question: 3x = 9, so x = 3",            # Correct
            "Question: 3x = 9, so x = 2"             # Incorrect
        ]
        labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        return texts, labels

    @staticmethod
    def load_gsm8k_data() -> Tuple[List[str], List[float]]:
        """
        Loads GSM8K dataset from Hugging Face and prepares correct and corrupted answers
        for verifier training.
        """
        try:
            print("Loading GSM8K for Verifier training...")
            dataset = hf_load_dataset("gsm8k", "main", split="train[:300]")
            texts = []
            labels = []
            
            for item in dataset:
                q = item["question"]
                a = item["answer"]
                
                # Correct answer sample (Label = 1.0)
                texts.append(f"Question: {q} Answer: {a}")
                labels.append(1.0)
                
                # Corrupted answer sample (Label = 0.0)
                corrupted_a = a.replace("The answer is", "The answer is not")
                texts.append(f"Question: {q} Answer: {corrupted_a}")
                labels.append(0.0)
                
            return texts, labels
        except Exception as e:
            print(f"Warning: Failed to load GSM8K for verifier: {e}. Using dummy verification data.")
            return VerifierTrainer.prepare_dummy_data()
