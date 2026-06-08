import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List

from harmony.config.settings import config
from harmony.chunking.semantic_chunker import SemanticChunker
from harmony.backbone.local_mixer import LocalMixer
from harmony.backbone.state_backbone import SelectiveStateBackbone
from harmony.generation.generator import GeneratorHead
from harmony.retrieval.fusion import RetrievalFusion
from harmony.retrieval.vector_memory import VectorMemory, Reranker, Retriever
from harmony.memory.memory_manager import MemoryManager
from harmony.memory.consolidator import MemoryConsolidator, MemoryEntry
from harmony.planner.planner_head import PlannerHead
from harmony.verifier.verifier_head import VerifierHead
from harmony.tokenizer.tokenizer_manager import TokenizerManager, TextEncoder, TextDecoder
from harmony.embeddings.embedding_manager import EmbeddingManager

class HarmonyModel(nn.Module):
    """
    Complete HARMONY-Core Pipeline.
    Integrates Tokenizer, EmbeddingManager, Ingestion, Memory Stack, Planner, Verifier, and Generator.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 2. Tokenizer, Encoder & Decoder
        self.tokenizer_manager = TokenizerManager(tokenizer_name=config.tokenizer_name)
        self.encoder = TextEncoder(self.tokenizer_manager)
        self.decoder = TextDecoder(self.tokenizer_manager)
        
        # 3. Chunker & Mixer
        self.chunker = SemanticChunker(tokenizer_name=config.tokenizer_name, chunk_size=config.chunk_size)
        self.local_mixer = LocalMixer(
            hidden_size=config.hidden_size,
            chunk_size=config.chunk_size,
            conv_kernel_size=config.conv_kernel_size
        )
        
        # 4. Selective State Backbone
        self.backbone = SelectiveStateBackbone(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
        
        # 5. Generator Head
        self.generator = GeneratorHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size
        )
        
        # Tie weights between embedding and generation head
        self.generator.proj.weight = self.embedding.weight
        
        # 6. Retrieval & Memory Systems
        self.embedding_manager = EmbeddingManager(model_name=config.embedding_model_name)
        self.memory_manager = MemoryManager(config.hidden_size, config.retrieval_dim)
        
        # Re-initialize VectorMemory and Retriever in memory_manager to use upgraded components
        self.memory_manager.semantic = VectorMemory(config.retrieval_dim, top_k=config.retrieval_top_k)
        self.reranker = Reranker()
        self.retriever = Retriever(self.memory_manager.semantic, self.reranker)
        
        self.memory_consolidator = MemoryConsolidator()
        self.working_memory: List[MemoryEntry] = []
        
        # 7. Control Modules (Planner, Verifier, Fusion)
        self.fusion = RetrievalFusion(
            hidden_size=config.hidden_size,
            retrieval_dim=config.retrieval_dim
        )
        self.planner = PlannerHead(hidden_size=config.hidden_size)
        self.verifier = VerifierHead(hidden_size=config.hidden_size)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expects chunked input ids: (batch, num_chunks, chunk_size)
        """
        B, C, S = input_ids.shape
        
        # Embed
        emb = self.embedding(input_ids) # (B, C, S, H)
        
        # Mix
        chunk_reprs = self.local_mixer(emb, attention_mask) # (B, C, H)
        
        # Backbone
        state_seq, new_state = self.backbone(chunk_reprs, state) # (B, C, H)
        
        # Generate
        logits = self.generator(state_seq) # (B, C, Vocab_size)
        
        return logits, new_state

    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        End-to-end inference pass for a raw string.
        """
        chunked_ids, mask = self.chunker.tokenize_and_chunk(text)
        
        device = next(self.parameters()).device
        chunked_ids = chunked_ids.to(device)
        mask = mask.to(device)
        
        logits, new_state = self.forward(chunked_ids, mask)
        
        return {
            "logits": logits,
            "final_state": new_state,
            "chunked_ids": chunked_ids
        }

    def generate_with_retry(
        self, 
        input_ids: torch.Tensor, 
        max_retries: int = 3, 
        threshold: float = 0.5,
        original_query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Phase B: Implements the confidence-based retry loop.
        Appends external documents or reasoning blocks if confidence score is low.
        """
        device = next(self.parameters()).device
        current_input_ids = input_ids.to(device)
        
        last_action = 0
        last_confidence = 0.0
        
        for attempt in range(max_retries):
            # Forward pass up to backbone
            logits, state = self.forward(current_input_ids)
            
            # The state sequence has shape (B, C, H). We take the final chunk's state for control.
            top_state = state[-1] # (Batch, Hidden)
            
            # 1. Planner Head (decides action)
            action_logits = self.planner(top_state)
            action = action_logits.argmax(dim=-1).item() # e.g. 0=generate, 1=retrieve, 2=reason, 3=verify, 4=stop
            last_action = action
            
            # 2. Verifier Head (scores confidence)
            confidence = self.verifier(top_state).item()
            last_confidence = confidence
            
            # 3. Stop or Accept criteria
            if confidence >= threshold or action == 4 or attempt == max_retries - 1:
                # Accept generation
                
                entry_text = f"Query: {original_query_text} | Attempt: {attempt}"
                meta = {"attempt": attempt, "final_action": action}
                new_entry = MemoryEntry(top_state[0], entry_text, meta, timestamp=time.time(), confidence=confidence)
                self.working_memory.append(new_entry)
                
                # Consolidate active working memories
                self.working_memory = self.memory_consolidator.consolidate(
                    self.working_memory,
                    top_state[0],
                    self.memory_manager.semantic,
                    self.memory_manager.episodic,
                    self.embedding_manager
                )
                
                return {
                    "action": action,
                    "confidence": confidence,
                    "logits": logits,
                    "retries": attempt,
                    "final_input_ids": current_input_ids
                }
            else:
                # Execution actions
                if action == 1:
                    # RETRIEVE action: query vector memory and append to inputs
                    query = original_query_text if original_query_text else "harmony"
                    q_emb = self.embedding_manager.get_query_embedding(query)
                    
                    # Fetch document
                    docs = self.retriever.retrieve(query, q_emb, k=1)
                    if docs:
                        retrieved_text = docs[0]["text"]
                        print(f"Retry Loop: Retrieved Context -> '{retrieved_text[:50]}...'")
                        
                        # Encode context and pad to chunk size
                        ctx_ids = self.encoder.encode(retrieved_text, return_tensors="pt").to(device)
                        
                        # Reshape and append to input_ids
                        ctx_chunks, _ = self.chunker.tokenize_and_chunk(retrieved_text)
                        ctx_chunks = ctx_chunks.to(device)
                        current_input_ids = torch.cat([current_input_ids, ctx_chunks], dim=1)
                        
                elif action == 2:
                    # REASON action: Append a "thinking" chunk (represented as padding/empty tokens)
                    print("Retry Loop: Executing REASON (thinking pause)...")
                    pad_chunk = torch.full((1, 1, config.chunk_size), self.tokenizer_manager.pad_token_id, dtype=torch.long, device=device)
                    current_input_ids = torch.cat([current_input_ids, pad_chunk], dim=1)
                    
                else:
                    # Other actions: verify, generate, etc. - fall back to standard loop iteration
                    pass
                    
        return {
            "action": last_action,
            "confidence": last_confidence,
            "logits": logits,
            "retries": max_retries - 1,
            "final_input_ids": current_input_ids
        }
