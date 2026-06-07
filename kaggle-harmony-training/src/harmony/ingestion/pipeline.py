import os
import hashlib
import json
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Optional/Required dependencies
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

from harmony.embeddings.embedding_manager import EmbeddingManager
from harmony.retrieval.vector_memory import VectorMemory

class KnowledgeIngestionPipeline:
    """
    Ingests text, markdown, PDF, and Word documents,
    chunks them, extracts sentence-transformer embeddings,
    and indexes them incrementally into the VectorMemory.
    """
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_memory: VectorMemory,
        state_db_path: str = "checkpoints/ingestion_state.json",
        chunk_size_char: int = 500,
        chunk_overlap_char: int = 100
    ):
        self.embedding_manager = embedding_manager
        self.vector_memory = vector_memory
        self.state_db_path = state_db_path
        self.chunk_size_char = chunk_size_char
        self.chunk_overlap_char = chunk_overlap_char
        
        # Load ingestion registry
        self.registry: Dict[str, str] = {}
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.state_db_path):
            try:
                with open(self.state_db_path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ingestion registry: {e}")
                self.registry = {}

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.state_db_path), exist_ok=True)
        with open(self.state_db_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def _get_file_hash(self, filepath: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def extract_text(self, filepath: str) -> str:
        """
        Parses document formats and returns raw text.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
                
        elif ext == ".pdf":
            if PdfReader is None:
                raise ImportError("pypdf is required to extract PDF files.")
            reader = PdfReader(filepath)
            text_parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
            return "\n\n".join(text_parts)
            
        elif ext == ".docx":
            if Document is None:
                raise ImportError("python-docx is required to extract Word docx files.")
            doc = Document(filepath)
            text_parts = [p.text for p in doc.paragraphs]
            return "\n".join(text_parts)
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of character length chunk_size_char, with overlap.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size_char
            chunk = text[start:end]
            chunks.append(chunk)
            
            start += (self.chunk_size_char - self.chunk_overlap_char)
            
        return chunks

    def ingest_file(self, filepath: str, force: bool = False) -> bool:
        """
        Ingests a single file into the vector database.
        Returns True if new content was indexed, False if skipped.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        file_hash = self._get_file_hash(filepath)
        filename = os.path.basename(filepath)
        
        # Check registry for incremental indexing
        if not force and self.registry.get(filepath) == file_hash:
            print(f"Skipping '{filename}' (already indexed and unchanged).")
            return False
            
        print(f"Ingesting '{filename}'...")
        
        # Extract, chunk, and embed
        text = self.extract_text(filepath)
        chunks = self.chunk_text(text)
        
        if not chunks:
            print(f"No text extracted from '{filename}'.")
            return False
            
        # If file was already indexed previously (but hash changed), clean old instances first
        if filepath in self.registry:
            self._purge_file_indices(filepath)
            
        # Get embeddings
        embeddings = self.embedding_manager.get_embeddings(chunks, show_progress=False)
        
        # Index chunks
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{filepath}#chunk_{idx}"
            metadata = {
                "source": filepath,
                "filename": filename,
                "chunk_idx": idx,
                "timestamp": os.path.getmtime(filepath)
            }
            self.vector_memory.add_document(doc_id, chunk, emb, metadata)
            
        # Register in DB
        self.registry[filepath] = file_hash
        self._save_registry()
        
        print(f"Successfully indexed {len(chunks)} chunks from '{filename}'.")
        return True

    def _purge_file_indices(self, filepath: str):
        """
        Deletes all index entries associated with a document path.
        """
        # Collect doc keys to delete
        keys_to_delete = []
        for key in list(self.vector_memory.doc_store.docs.keys()):
            if key.startswith(f"{filepath}#"):
                keys_to_delete.append(key)
                
        for key in keys_to_delete:
            self.vector_memory.remove_document(key)

    def ingest_directory(self, directory: str, recursive: bool = True, force: bool = False):
        """
        Ingests all supported files (.txt, .md, .pdf, .docx) inside a directory.
        """
        print(f"Scanning directory: {directory}...")
        supported_exts = {".txt", ".md", ".pdf", ".docx"}
        files_to_ingest = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_exts:
                        files_to_ingest.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                filepath = os.path.join(directory, file)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in supported_exts:
                        files_to_ingest.append(filepath)
                        
        print(f"Found {len(files_to_ingest)} candidate file(s) for ingestion.")
        
        new_count = 0
        for filepath in tqdm(files_to_ingest, desc="Ingesting files"):
            try:
                if self.ingest_file(filepath, force=force):
                    new_count += 1
            except Exception as e:
                print(f"Error ingesting '{filepath}': {e}")
                
        print(f"Ingestion directory pass complete. Newly indexed files: {new_count}/{len(files_to_ingest)}")
