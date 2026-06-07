import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset
from typing import Optional, Union, Dict, Any, List
from harmony.tokenizer.tokenizer_manager import TokenizerManager

class HarmonyDataset(Dataset):
    """
    Custom PyTorch Dataset that prepares tokenized text into sliding windows
    for the chunk-based HARMONY model.
    Each sample returns:
      - input_ids: (num_chunks, chunk_size)
      - target_ids: (num_chunks,) -> target token immediately following each chunk
    """
    def __init__(self, token_ids: List[int], chunk_size: int, num_chunks: int, stride: Optional[int] = None):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.window_size = num_chunks * chunk_size
        self.total_seq_len = self.window_size + 1
        
        # Stride defaults to window_size (non-overlapping)
        self.stride = stride if stride is not None else self.window_size
        
        # Convert to tensor for fast slicing
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        
        # Calculate number of samples
        if len(self.tokens) < self.total_seq_len:
            self.num_samples = 0
        else:
            self.num_samples = (len(self.tokens) - self.total_seq_len) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.total_seq_len
        
        window = self.tokens[start_idx:end_idx]
        
        # Input ids are the first window_size tokens (0 to window_size-1)
        input_ids_flat = window[:self.window_size]
        input_ids = input_ids_flat.view(self.num_chunks, self.chunk_size)
        
        # Targets are the tokens immediately following each chunk.
        # e.g., for chunk i (from i*S to (i+1)*S - 1), target is token at (i+1)*S.
        # Thus, target indices are S, 2S, 3S, ..., num_chunks*S
        target_indices = torch.arange(1, self.num_chunks + 1) * self.chunk_size
        target_ids = window[target_indices]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }


class DatasetManager:
    """
    Orchestrates downloading, tokenizing, caching, and building dataloaders
    for training datasets (WikiText, OpenWebText, or local files).
    """
    def __init__(self, tokenizer_manager: TokenizerManager, chunk_size: int = 16, num_chunks: int = 32):
        self.tokenizer_manager = tokenizer_manager
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        
    def load_hf_dataset(self, path: str, name: Optional[str] = None, split: str = "train") -> List[int]:
        """
        Loads a Hugging Face dataset, tokenizes all text, and returns a flat list of token IDs.
        """
        print(f"Loading HF dataset '{path}' (name={name}, split={split})...")
        dataset = hf_load_dataset(path, name, split=split)
        
        # Extract text field (standard fields are 'text' or 'content')
        text_field = "text" if "text" in dataset.features else None
        if text_field is None:
            for field in ["content", "document", "story"]:
                if field in dataset.features:
                    text_field = field
                    break
            if text_field is None:
                raise ValueError(f"Could not find text or content field in dataset features: {dataset.features.keys()}")
                
        tokenizer = self.tokenizer_manager.get_tokenizer()
        
        print("Tokenizing dataset (this might take a few moments)...")
        # Tokenize in batches using datasets' fast mapping
        def tokenize_func(examples):
            return tokenizer(examples[text_field], truncation=False, padding=False)
            
        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Flatten token IDs list
        flat_token_ids = []
        for item in tokenized_dataset:
            flat_token_ids.extend(item["input_ids"])
            
        print(f"Finished tokenization. Total tokens: {len(flat_token_ids)}")
        return flat_token_ids

    def load_local_dataset(self, file_path: str) -> List[int]:
        """
        Loads a local raw text file, tokenizes it, and returns a flat list of token IDs.
        """
        print(f"Loading local file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        tokenizer = self.tokenizer_manager.get_tokenizer()
        tokenized = tokenizer(text, truncation=False, padding=False)
        flat_token_ids = tokenized["input_ids"]
        
        print(f"Finished tokenization. Total tokens: {len(flat_token_ids)}")
        return flat_token_ids

    def build_dataloader(
        self, 
        token_ids: List[int], 
        batch_size: int = 8, 
        shuffle: bool = True, 
        stride: Optional[int] = None,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Builds a standard PyTorch DataLoader wrapped around HarmonyDataset.
        """
        dataset = HarmonyDataset(
            token_ids=token_ids,
            chunk_size=self.chunk_size,
            num_chunks=self.num_chunks,
            stride=stride
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        return loader
