import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Optional

class HarmonyEvaluator:
    """
    Automated benchmark suite for evaluating HARMONY performance across:
      - Language Modeling (Perplexity)
      - Retrieval (Recall@k, MRR)
      - Verification (Accuracy, F1 Score)
      - Planning (Action Choice Accuracy)
      - System Performance (Latency, Memory usage)
    """
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def evaluate_language(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Computes language modeling loss and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        total_elements = 0
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                targets = batch["target_ids"].to(self.device)
                
                # Forward pass
                # inputs shape: (batch, num_chunks, chunk_size)
                # logits shape: (batch, num_chunks, vocab_size)
                logits, _ = self.model(inputs)
                
                # Reshape for cross entropy loss
                vocab_size = logits.size(-1)
                loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
                
                total_loss += loss.item()
                total_elements += targets.numel()
                
        avg_loss = total_loss / max(1, total_elements)
        # Clamp to prevent math overflow on untrained / high-loss models
        perplexity = math.exp(min(avg_loss, 100)) if avg_loss < 100 else float('inf')
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }

    def evaluate_retrieval(
        self, 
        vector_memory: Any, 
        embedding_manager: Any, 
        queries: List[str], 
        ground_truth: List[str], 
        k: int = 3
    ) -> Dict[str, float]:
        """
        Evaluates retrieval Recall@k and Mean Reciprocal Rank (MRR).
        """
        hits = 0
        mrr_sum = 0.0
        total = len(queries)
        
        if total == 0:
            return {"recall_at_k": 0.0, "mrr": 0.0}
            
        for query, gt_doc_id in zip(queries, ground_truth):
            q_emb = embedding_manager.get_query_embedding(query)
            
            # Retrieve top k documents
            _, results = vector_memory.search(q_emb.unsqueeze(0), k=k)
            retrieved_docs = results[0] # first query in batch
            
            # Check Recall and MRR
            rank = -1
            for rank_idx, doc in enumerate(retrieved_docs):
                if doc["id"] == gt_doc_id:
                    rank = rank_idx + 1
                    break
                    
            if rank != -1:
                hits += 1
                mrr_sum += 1.0 / rank
                
        return {
            "recall_at_k": hits / total,
            "mrr": mrr_sum / total
        }

    def evaluate_verifier(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates verifier head accuracy, precision, recall, and F1.
        """
        self.model.eval()
        tp = fp = fn = tn = 0
        
        with torch.no_grad():
            for states, labels in dataloader:
                states = states.to(self.device)
                labels = labels.to(self.device).view(-1)
                
                # Verifier forward
                # self.model.verifier returns (batch, 1) confidence score
                confidence = self.model.verifier(states).view(-1)
                predictions = (confidence >= 0.5).float()
                
                for pred, label in zip(predictions, labels):
                    p_val = int(pred.item())
                    l_val = int(label.item())
                    
                    if p_val == 1 and l_val == 1:
                        tp += 1
                    elif p_val == 1 and l_val == 0:
                        fp += 1
                    elif p_val == 0 and l_val == 1:
                        fn += 1
                    else:
                        tn += 1
                        
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / max(1, total)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_score = 2 * (precision * recall) / max(1e-8, precision + recall)
        
        return {
            "verifier_accuracy": accuracy,
            "verifier_precision": precision,
            "verifier_recall": recall,
            "verifier_f1": f1_score
        }

    def evaluate_planner(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluates planner head action selection accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, targets in dataloader:
                states = states.to(self.device)
                targets = targets.to(self.device).view(-1)
                
                logits = self.model.planner(states) # (batch, num_actions)
                predictions = logits.argmax(dim=-1)
                
                correct += (predictions == targets).sum().item()
                total += targets.numel()
                
        return {
            "planner_accuracy": correct / max(1, total)
        }

    def benchmark_performance(self, sample_text: str, num_runs: int = 10) -> Dict[str, Any]:
        """
        Measures inference latency and memory footprints.
        """
        self.model.eval()
        
        # Reset memory tracking
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            
        t_start = time.perf_counter()
        
        # Warmup
        with torch.no_grad():
            self.model.process_text(sample_text)
            
        t_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                self.model.process_text(sample_text)
            t_latencies.append((time.perf_counter() - t0) * 1000.0)
            
        avg_latency_ms = sum(t_latencies) / num_runs
        
        # Peak memory
        if self.device == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # CPU peak memory (simple placeholder or approximation)
            peak_memory_mb = 0.0
            
        return {
            "avg_latency_ms": avg_latency_ms,
            "peak_memory_mb": peak_memory_mb,
            "num_runs": num_runs
        }
