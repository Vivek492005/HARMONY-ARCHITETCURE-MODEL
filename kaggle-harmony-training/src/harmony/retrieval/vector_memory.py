import os
import time
import json
import pickle
import faiss
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

class DocumentStore:
    """
    In-memory storage for document texts and metadata.
    """
    def __init__(self):
        self.docs: Dict[str, Dict[str, Any]] = {}

    def add(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        self.docs[doc_id] = {
            "text": text,
            "metadata": metadata
        }

    def remove(self, doc_id: str):
        if doc_id in self.docs:
            del self.docs[doc_id]

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.docs.get(doc_id)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, indent=2)

    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.docs = json.load(f)


class Reranker:
    """
    Computes dot-product similarity between query and candidate document embeddings
    to rerank top candidates.
    """
    def __init__(self):
        pass

    def rerank(self, query_emb: torch.Tensor, candidate_embs: torch.Tensor, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks candidates based on cosine similarity of embeddings.
        """
        if not candidates or len(candidates) == 0:
            return []
            
        # Compute cosine similarity
        # query_emb: (hidden_size,) or (1, hidden_size)
        # candidate_embs: (num_candidates, hidden_size)
        q = query_emb.view(1, -1)
        c = candidate_embs
        
        # Normalize
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        c_norm = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
        
        scores = torch.mm(q_norm, c_norm.transpose(0, 1)).squeeze(0) # (num_candidates,)
        
        # Add score to each candidate
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score.item())
            
        # Sort descending by score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates


class VectorMemory:
    """
    Upgraded VectorMemory using FAISS FlatL2 index with full document CRUD operations,
    metadata filtering, persistence, and latency tracking.
    """
    def __init__(self, retrieval_dim: int, top_k: int = 3):
        self.retrieval_dim = retrieval_dim
        self.top_k = top_k
        self.index = faiss.IndexFlatL2(retrieval_dim)
        self.doc_store = DocumentStore()
        
        # Map FAISS index integer ID -> doc_id (str)
        self.faiss_to_doc_id: Dict[int, str] = {}
        # Keep track of active embeddings in numpy array form to enable index rebuilding on deletion
        self.doc_id_to_embedding: Dict[str, np.ndarray] = {}
        
        # Latency metrics
        self.last_query_latency_ms: float = 0.0

    def add_document(self, doc_id: str, text: str, embedding: torch.Tensor, metadata: Dict[str, Any]):
        """
        Adds a single document text, embedding, and metadata to storage and index.
        """
        emb_np = embedding.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
        assert emb_np.shape[1] == self.retrieval_dim, f"Embedding dim {emb_np.shape[1]} doesn't match {self.retrieval_dim}"
        
        # Add to document store
        self.doc_store.add(doc_id, text, metadata)
        
        # Keep embedding
        self.doc_id_to_embedding[doc_id] = emb_np
        
        # Rebuild index
        self._rebuild_index()

    def remove_document(self, doc_id: str):
        """
        Removes a document from stores and rebuilds the FAISS index.
        """
        if doc_id in self.doc_store.docs:
            self.doc_store.remove(doc_id)
            
        if doc_id in self.doc_id_to_embedding:
            del self.doc_id_to_embedding[doc_id]
            
        self._rebuild_index()

    def update_document(self, doc_id: str, text: str, embedding: torch.Tensor, metadata: Dict[str, Any]):
        """
        Updates document text, embedding, and metadata.
        """
        self.add_document(doc_id, text, embedding, metadata)

    def _rebuild_index(self):
        """
        Rebuilds the FAISS index from the currently active embeddings.
        """
        self.index = faiss.IndexFlatL2(self.retrieval_dim)
        self.faiss_to_doc_id.clear()
        
        if not self.doc_id_to_embedding:
            return
            
        # Collect all active embeddings and map them
        embeddings_list = []
        for idx, (doc_id, emb) in enumerate(self.doc_id_to_embedding.items()):
            embeddings_list.append(emb)
            self.faiss_to_doc_id[idx] = doc_id
            
        all_embs = np.vstack(embeddings_list)
        self.index.add(all_embs)

    def search(
        self, 
        queries: torch.Tensor, 
        k: Optional[int] = None, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]]]:
        """
        Retrieves matching documents, matching query embeddings against index.
        Supports query batches.
        Args:
            queries: Tensor of shape (batch, retrieval_dim)
            k: Optional top-k limit.
            metadata_filter: dictionary of key-value matches.
        Returns:
            distances: distances tensor of shape (batch, top_k)
            retrieved_docs: List of lists of document dicts containing ID, text, metadata.
        """
        t0 = time.perf_counter()
        
        if k is None:
            k = self.top_k
            
        total_items = self.index.ntotal
        if total_items == 0:
            self.last_query_latency_ms = (time.perf_counter() - t0) * 1000.0
            return torch.empty((queries.size(0), 0)), [[] for _ in range(queries.size(0))]
            
        # Search for more than K items initially to allow metadata filtering
        search_k = min(total_items, k * 4 if metadata_filter else k)
        
        q_np = queries.detach().cpu().numpy().astype(np.float32)
        distances, indices = self.index.search(q_np, search_k)
        
        batch_results = []
        batch_distances = []
        
        for batch_idx, row in enumerate(indices):
            docs_row = []
            dists_row = []
            
            for idx_in_row, val in enumerate(row):
                if val == -1:
                    continue
                    
                doc_id = self.faiss_to_doc_id.get(val)
                if not doc_id:
                    continue
                    
                doc = self.doc_store.get(doc_id)
                if not doc:
                    continue
                    
                # Apply metadata filter if specified
                if metadata_filter:
                    match = True
                    doc_meta = doc.get("metadata", {})
                    for fk, fv in metadata_filter.items():
                        if doc_meta.get(fk) != fv:
                            match = False
                            break
                    if not match:
                        continue
                        
                docs_row.append({
                    "id": doc_id,
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "embedding": torch.tensor(self.doc_id_to_embedding[doc_id])
                })
                dists_row.append(distances[batch_idx, idx_in_row])
                
                # Check limit
                if len(docs_row) == k:
                    break
                    
            batch_results.append(docs_row)
            # Pad distance list to match doc list
            batch_distances.append(dists_row + [float('inf')] * (k - len(dists_row)))
            
        self.last_query_latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # Convert distances back to PyTorch
        distances_pt = torch.tensor(batch_distances, device=queries.device)
        return distances_pt, batch_results

    def save(self, directory: str):
        """
        Saves the index, metadata, and doc store to disk.
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save mappings & embeddings
        with open(os.path.join(directory, "embeddings.pkl"), "wb") as f:
            pickle.dump({
                "faiss_to_doc_id": self.faiss_to_doc_id,
                "doc_id_to_embedding": self.doc_id_to_embedding
            }, f)
            
        # Save document store
        self.doc_store.save(os.path.join(directory, "documents.json"))

    def load(self, directory: str):
        """
        Loads index, metadata, and doc store from disk.
        """
        index_path = os.path.join(directory, "index.faiss")
        embed_path = os.path.join(directory, "embeddings.pkl")
        doc_path = os.path.join(directory, "documents.json")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
        if os.path.exists(embed_path):
            with open(embed_path, "rb") as f:
                data = pickle.load(f)
                self.faiss_to_doc_id = data.get("faiss_to_doc_id", {})
                self.doc_id_to_embedding = data.get("doc_id_to_embedding", {})
                
        if os.path.exists(doc_path):
            self.doc_store.load(doc_path)


class Retriever:
    """
    Connects VectorMemory and Reranker to execute retrieval and ranking pipeline.
    """
    def __init__(self, vector_memory: VectorMemory, reranker: Reranker):
        self.memory = vector_memory
        self.reranker = reranker

    def retrieve(
        self, 
        query_text: str, 
        query_embedding: torch.Tensor, 
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs retrieval + reranking.
        """
        # Search top items (search twice as many for reranking)
        search_k = k if k is not None else self.memory.top_k
        
        # Wrap query_embedding in batch dimension if needed
        if query_embedding.ndim == 1:
            q_batch = query_embedding.unsqueeze(0)
        else:
            q_batch = query_embedding
            
        _, results = self.memory.search(q_batch, k=search_k * 2, metadata_filter=metadata_filter)
        candidates = results[0] # first query in batch
        
        if not candidates:
            return []
            
        # Stack candidate embeddings
        cand_embs = torch.cat([c["embedding"].view(1, -1) for c in candidates], dim=0)
        
        # Rerank
        reranked = self.reranker.rerank(query_embedding, cand_embs, candidates)
        
        # Return top K
        return reranked[:search_k]
