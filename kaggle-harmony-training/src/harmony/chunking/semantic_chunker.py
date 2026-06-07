import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from transformers import AutoTokenizer

class SemanticChunker(nn.Module):
    """
    Splits input into semantic chunks.
    For v1 MVP, we use fixed-size chunking after subword tokenization.
    """
    def __init__(self, tokenizer_name: str = "gpt2", chunk_size: int = 16):
        super().__init__()
        self.chunk_size = chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def tokenize_and_chunk(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes text and returns padded chunks.
        Returns:
            chunked_input_ids: (batch=1, num_chunks, chunk_size)
            attention_mask: (batch=1, num_chunks, chunk_size)
        """
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"] # (1, seq_len)
        
        seq_len = input_ids.size(1)
        # Pad to multiple of chunk_size
        remainder = seq_len % self.chunk_size
        if remainder > 0:
            pad_len = self.chunk_size - remainder
            pad_tensor = torch.full((1, pad_len), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_tensor], dim=1)
            
        num_chunks = input_ids.size(1) // self.chunk_size
        chunked_ids = input_ids.view(1, num_chunks, self.chunk_size)
        
        # Simple attention mask
        attention_mask = (chunked_ids != self.tokenizer.pad_token_id).long()
        
        return chunked_ids, attention_mask
