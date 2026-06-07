import pytest
import torch
from harmony.chunking.semantic_chunker import SemanticChunker

def test_semantic_chunker():
    chunk_size = 8
    chunker = SemanticChunker(tokenizer_name="gpt2", chunk_size=chunk_size)
    
    # Test text
    text = "HARMONY semantic chunking system unit test."
    chunked_ids, attention_mask = chunker.tokenize_and_chunk(text)
    
    # Assertions
    assert isinstance(chunked_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert chunked_ids.ndim == 3  # (batch=1, num_chunks, chunk_size)
    assert chunked_ids.shape[0] == 1
    assert chunked_ids.shape[2] == chunk_size
    assert attention_mask.shape == chunked_ids.shape
    
    # Ensure attention mask has binary elements
    assert torch.all((attention_mask == 0) | (attention_mask == 1))
    
    # Ensure pad token is padded correctly
    pad_id = chunker.tokenizer.pad_token_id
    if pad_id is not None:
        last_chunk = chunked_ids[0, -1]
        last_mask = attention_mask[0, -1]
        # For elements where it is pad_id, mask should be 0
        for i in range(chunk_size):
            if last_chunk[i] == pad_id:
                assert last_mask[i] == 0
