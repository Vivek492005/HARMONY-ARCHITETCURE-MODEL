from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class HarmonyConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HARMONY_")

    # Model architecture params
    vocab_size: int = 50257 # GPT-2 default
    hidden_size: int = 256
    num_layers: int = 4
    
    # Chunking params
    chunk_size: int = 16
    
    # Local mixer params
    conv_kernel_size: int = 3
    
    # Memory params
    retrieval_dim: int = 384
    retrieval_top_k: int = 3
    
    # Embedding params
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Tokenizer params
    tokenizer_name: str = "gpt2"
    
    # Ingestion params
    chunk_size_char: int = 500
    chunk_overlap_char: int = 100
    state_db_path: str = "checkpoints/ingestion_state.json"
    
    # Training params
    batch_size: int = 8
    epochs_stage1: int = 3
    epochs_stage3: int = 3
    epochs_stage4: int = 10
    epochs_stage5: int = 3
    learning_rate: float = 1e-4
    checkpoint_dir: str = "checkpoints"

config = HarmonyConfig()
