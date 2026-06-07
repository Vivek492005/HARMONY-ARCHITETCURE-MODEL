import os
import torch
from torch.utils.data import DataLoader
from harmony.models.harmony_model import HarmonyModel
from harmony.training.orchestrator import TrainingOrchestrator
from harmony.data.dataset_manager import DatasetManager
from harmony.ingestion.pipeline import KnowledgeIngestionPipeline
from harmony.planner.training import PlannerDataset, PlannerTrainer
from harmony.verifier.training import VerifierDataset, VerifierTrainer
from harmony.config.settings import config

def main():
    print("=== HARMONY-Core Full 5-Stage Curriculum Training Pipeline ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize model and orchestrator
    model = HarmonyModel()
    orchestrator = TrainingOrchestrator(model, device=device, checkpoint_dir=config.checkpoint_dir)
    
    # --- STAGE 1: Backbone Pretraining ---
    print("\n--- [Stage 1] Preparing Language Dataset ---")
    dm = DatasetManager(model.tokenizer_manager, chunk_size=config.chunk_size, num_chunks=4)
    
    # Try loading local dataset from workspace
    workspace_dir = ".."
    report_path = os.path.join(workspace_dir, "deep-research-report.md")
    
    if os.path.exists(report_path):
        print(f"Loading real text data from {report_path} for pretraining...")
        token_ids = dm.load_local_dataset(report_path)
    elif os.path.exists("deep-research-report.md"):
        print("Loading real text data from deep-research-report.md for pretraining...")
        token_ids = dm.load_local_dataset("deep-research-report.md")
    else:
        print("No local report found, generating dummy tokens for pretraining...")
        token_ids = list(range(2000))
        
    # Build loader
    lm_loader = dm.build_dataloader(token_ids, batch_size=2, shuffle=True)
    
    # Run Stage 1 training
    orchestrator.train_stage_1_backbone(lm_loader, epochs=config.epochs_stage1, lr=1e-4)
    
    # --- STAGE 2: Retrieval Memory Setup ---
    print("\n--- [Stage 2] Setting up Retrieval Memory (Ingestion) ---")

    vector_memory_dir = os.path.join(config.checkpoint_dir, "vector_memory")

    # Try to load an existing (non-empty) VectorMemory index first
    index_path = os.path.join(vector_memory_dir, "index.faiss")
    if os.path.exists(index_path):
        model.memory_manager.semantic.load(vector_memory_dir)
        n_existing = model.memory_manager.semantic.index.ntotal
        print(f"Loaded existing VectorMemory: {n_existing} vectors.")
    else:
        n_existing = 0

    # Initialize ingestion pipeline
    pipeline = KnowledgeIngestionPipeline(
        embedding_manager=model.embedding_manager,
        vector_memory=model.memory_manager.semantic,
        state_db_path=config.state_db_path,
        chunk_size_char=config.chunk_size_char,
        chunk_overlap_char=config.chunk_overlap_char
    )

    # If VectorMemory is empty the registry is stale — wipe it so files get re-indexed
    if model.memory_manager.semantic.index.ntotal == 0:
        print("VectorMemory index is empty. Clearing stale ingestion registry to force re-ingestion...")
        pipeline.registry = {}
        pipeline._save_registry()

    # Ingest workspace documents
    docs_to_ingest = [
        "deep-research-report.md",
        "HARMONY FINAL.pdf",
        "HARMONY RESEARCH PAPER.pdf",
        "HARMONY_Cognitive_Architecture.pdf"
    ]

    ingested_any = False
    for doc in docs_to_ingest:
        doc_path = os.path.join(workspace_dir, doc)
        if not os.path.exists(doc_path):
            doc_path = doc  # fallback: relative to cwd
        if os.path.exists(doc_path):
            try:
                result = pipeline.ingest_file(doc_path)
                if result:
                    ingested_any = True
            except Exception as e:
                print(f"Error ingesting {doc}: {e}")
        else:
            print(f"Document not found, skipping: {doc}")

    if not ingested_any and model.memory_manager.semantic.index.ntotal == 0:
        print("Warning: No workspace files ingested. Seeding VectorMemory with mock vectors...")
        for i in range(10):
            doc_id = f"mock_doc_{i}"
            text = (
                f"Harmony cognitive architecture document {i}: "
                "The HARMONY model uses a selective state space backbone "
                "with planner, verifier, and retrieval heads for general reasoning."
            )
            emb = torch.randn(config.retrieval_dim)
            model.memory_manager.semantic.add_document(doc_id, text, emb, {"topic": "harmony"})

    n_total = model.memory_manager.semantic.index.ntotal
    print(f"Total vectors in VectorMemory: {n_total}")

    # Persist VectorMemory so eval.py can load it
    model.memory_manager.semantic.save(vector_memory_dir)
    print(f"VectorMemory saved to {vector_memory_dir}")
    
    # --- STAGE 3: Planner Training ---
    print("\n--- [Stage 3] Training Planner Head ---")
    p_questions, p_actions = PlannerTrainer.prepare_dummy_data()
    planner_dataset = PlannerDataset(p_questions, p_actions, model)
    planner_loader = DataLoader(planner_dataset, batch_size=2, shuffle=True)
    
    orchestrator.train_stage_3_planner(planner_loader, epochs=config.epochs_stage3, lr=1e-4)
    
    # --- STAGE 4: Verifier Training ---
    print("\n--- [Stage 4] Training Verifier Head ---")
    v_texts, v_labels = VerifierTrainer.load_gsm8k_data()
    verifier_dataset = VerifierDataset(v_texts, v_labels, model)
    verifier_loader = DataLoader(verifier_dataset, batch_size=2, shuffle=True)
    
    orchestrator.train_stage_4_verifier(verifier_loader, epochs=config.epochs_stage4, lr=1e-4)
    
    # --- STAGE 5: Joint Fine-Tuning ---
    print("\n--- [Stage 5] Joint Fine-Tuning ---")
    orchestrator.train_stage_5_joint(
        lm_loader=lm_loader,
        planner_loader=planner_loader,
        verifier_loader=verifier_loader,
        epochs=config.epochs_stage5,
        lr=1e-5
    )
    
    # Save final full model state
    final_ckpt_path = os.path.join(config.checkpoint_dir, "harmony_final.pt")
    torch.save({"model_state_dict": model.state_dict()}, final_ckpt_path)
    print(f"\nFinal model checkpoint saved to: {final_ckpt_path}")
    print("\n=== All 5 Stages of HARMONY-Core Curriculum completed successfully! ===")

if __name__ == "__main__":
    main()
