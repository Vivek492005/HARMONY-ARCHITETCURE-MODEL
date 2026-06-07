import sys
import types
import torch
import numpy as np

# Track the last retrieval dimension to dynamically scale mock embeddings
LAST_RETRIEVAL_DIM = 384

# Mock VectorMemory.__init__ to capture the retrieval dimension
try:
    from harmony.retrieval.vector_memory import VectorMemory
    original_vm_init = VectorMemory.__init__
    
    def mocked_vm_init(self, retrieval_dim, *args, **kwargs):
        global LAST_RETRIEVAL_DIM
        LAST_RETRIEVAL_DIM = retrieval_dim
        original_vm_init(self, retrieval_dim, *args, **kwargs)
        
    VectorMemory.__init__ = mocked_vm_init
except Exception as e:
    print(f"conftest.py warning: Could not mock VectorMemory init: {e}")


# Mock SentenceTransformer
class MockSentenceTransformer:
    def __init__(self, model_name=None, device=None, **kwargs):
        self.model_name = model_name
        self.device = device

    def encode(self, sentences, batch_size=32, convert_to_tensor=False, device=None, show_progress_bar=False, **kwargs):
        global LAST_RETRIEVAL_DIM
        dim = LAST_RETRIEVAL_DIM
        
        # Handle single string query
        if isinstance(sentences, str):
            res = np.zeros(dim, dtype=np.float32)
            if convert_to_tensor:
                return torch.from_numpy(res).to(device or "cpu")
            return res
            
        # Handle list of strings
        res = np.zeros((len(sentences), dim), dtype=np.float32)
        if convert_to_tensor:
            return torch.from_numpy(res).to(device or "cpu")
        return res

    def get_sentence_embedding_dimension(self):
        global LAST_RETRIEVAL_DIM
        return LAST_RETRIEVAL_DIM


# Inject mock sentence_transformers module
st_mock = types.ModuleType("sentence_transformers")
st_mock.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = st_mock


# Mock Hugging Face datasets to fail immediately and trigger fallback logic
def mocked_load_dataset(*args, **kwargs):
    raise ConnectionError("Mocked connection error for offline testing")

datasets_mock = types.ModuleType("datasets")
datasets_mock.load_dataset = mocked_load_dataset
sys.modules["datasets"] = datasets_mock
