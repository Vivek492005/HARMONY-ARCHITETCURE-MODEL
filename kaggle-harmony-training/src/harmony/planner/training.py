import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from datasets import load_dataset as hf_load_dataset

from harmony.tokenizer.tokenizer_manager import TokenizerManager
from harmony.models.harmony_model import HarmonyModel

class PlannerDataset(Dataset):
    """
    Prepares reasoning steps and maps them to target planner actions.
    Actions: 0=generate, 1=retrieve, 2=reason, 3=verify, 4=stop
    """
    def __init__(self, questions: List[str], actions: List[int], model: HarmonyModel):
        self.questions = questions
        self.actions = actions
        self.model = model
        self.model.eval()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        question = self.questions[idx]
        action = self.actions[idx]
        
        # Run backbone forward pass to get state representation
        with torch.no_grad():
            res = self.model.process_text(question)
            # final_state is (num_layers, batch=1, hidden_size)
            # Take top layer, remove batch dim -> (hidden_size,)
            state = res["final_state"][-1].squeeze(0)
            
        return state, action


class PlannerTrainer:
    """
    Trains the PlannerHead of the HarmonyModel using supervised learning.
    """
    def __init__(self, model: HarmonyModel, lr: float = 1e-4, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.planner.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for states, targets in dataloader:
            states = states.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model.planner(states) # (batch, num_actions)
            loss = self.loss_fn(logits, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    @staticmethod
    def prepare_dummy_data() -> Tuple[List[str], List[int]]:
        """
        Fallback data for bootstrapping training without internet.
        """
        questions = [
            "What is 2 + 2?", 
            "Calculate the derivative of x^2.", 
            "Who was the first president of the United States?",
            "What is the capital of France?",
            "Solve for x: 3x - 9 = 0",
            "Write a python function to reverse a string."
        ]
        # Actions: 0=generate, 1=retrieve, 2=reason, 3=verify, 4=stop
        actions = [0, 2, 1, 1, 2, 0]
        return questions, actions

    @staticmethod
    def load_gsm8k_data() -> Tuple[List[str], List[int]]:
        """
        Loads GSM8K dataset from Hugging Face and returns question-action pairs.
        """
        try:
            print("Loading GSM8K for Planner training...")
            dataset = hf_load_dataset("gsm8k", "main", split="train[:500]")
            questions = []
            actions = []
            
            for item in dataset:
                q = item["question"]
                questions.append(q)
                # Mathematical questions require REASON (2) action
                actions.append(2)
                
            return questions, actions
        except Exception as e:
            print(f"Warning: Failed to load GSM8K: {e}. Using dummy reasoning data.")
            return PlannerTrainer.prepare_dummy_data()
