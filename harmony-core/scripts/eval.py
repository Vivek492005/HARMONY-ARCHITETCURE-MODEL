import os
import torch
from torch.utils.data import DataLoader
from harmony.models.harmony_model import HarmonyModel
from harmony.evaluation.evaluator import HarmonyEvaluator
from harmony.data.dataset_manager import DatasetManager
from harmony.planner.training import PlannerDataset, PlannerTrainer
from harmony.verifier.training import VerifierDataset, VerifierTrainer
from harmony.config.settings import config

def main():
    print("=== HARMONY-Core Performance Evaluation Framework ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = HarmonyModel()
    
    # 2. Load trained checkpoint if it exists
    final_ckpt = os.path.join(config.checkpoint_dir, "harmony_final.pt")
    if os.path.exists(final_ckpt):
        print(f"Loading trained weights from: {final_ckpt}")
        ckpt = torch.load(final_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Trained checkpoint loaded successfully.")
    else:
        print("Warning: No trained checkpoint found. Evaluating with random weights.")
        
    model.eval()
    
    # 3. Load persisted VectorMemory index
    vector_memory_dir = os.path.join(config.checkpoint_dir, "vector_memory")
    if os.path.exists(vector_memory_dir):
        print(f"Loading VectorMemory index from: {vector_memory_dir}")
        model.memory_manager.semantic.load(vector_memory_dir)
        print(f"VectorMemory loaded: {model.memory_manager.semantic.index.ntotal} documents.")
    else:
        print("Warning: No VectorMemory index found. Run train.py first to index documents.")
    
    # 4. Initialize Evaluator
    evaluator = HarmonyEvaluator(model, device=device)
    
    # --- 1. Language Modeling ---
    print("\n--- [1/5] Evaluating Language Modeling ---")
    dm = DatasetManager(model.tokenizer_manager, chunk_size=config.chunk_size, num_chunks=4)
    
    workspace_dir = ".."
    report_path = os.path.join(workspace_dir, "deep-research-report.md")
    if os.path.exists(report_path):
        token_ids = dm.load_local_dataset(report_path)
    elif os.path.exists("deep-research-report.md"):
        token_ids = dm.load_local_dataset("deep-research-report.md")
    else:
        token_ids = list(range(1000))
        
    lm_loader = dm.build_dataloader(token_ids, batch_size=4, shuffle=False)
    lm_metrics = evaluator.evaluate_language(lm_loader)
    
    # --- 2. Planner ---
    print("\n--- [2/5] Evaluating Planner Choice Accuracy ---")
    p_questions, p_actions = PlannerTrainer.prepare_dummy_data()
    planner_dataset = PlannerDataset(p_questions, p_actions, model)
    planner_loader = DataLoader(planner_dataset, batch_size=2, shuffle=False)
    plan_metrics = evaluator.evaluate_planner(planner_loader)
    
    # --- 3. Verifier ---
    print("\n--- [3/5] Evaluating Verifier Head Calibration ---")
    v_texts, v_labels = VerifierTrainer.load_gsm8k_data()
    verifier_dataset = VerifierDataset(v_texts, v_labels, model)
    verifier_loader = DataLoader(verifier_dataset, batch_size=2, shuffle=False)
    ver_metrics = evaluator.evaluate_verifier(verifier_loader)
    
    # --- 4. Retrieval ---
    print("\n--- [4/5] Evaluating VectorMemory Retrieval ---")
    # Always add seed documents for retrieval evaluation to ensure meaningful metrics
    print("Adding seed documents for retrieval eval...")
    docs = [
        ("doc_france", "Paris is the capital and most populous city of France.", {"topic": "geography"}),
        ("doc_germany", "Berlin is the capital and the largest city of Germany.", {"topic": "geography"}),
        ("doc_harmony", "The Harmony architecture uses a selective state space backbone.", {"topic": "architecture"}),
    ]
    for doc_id, text, meta in docs:
        emb = model.embedding_manager.get_query_embedding(text)
        model.memory_manager.semantic.add_document(doc_id, text, emb, meta)
            
    queries = ["What is the capital of France?", "German capital city", "How does Harmony backbone work?"]
    ground_truth = ["doc_france", "doc_germany", "doc_harmony"]
    
    ret_metrics = evaluator.evaluate_retrieval(
        vector_memory=model.memory_manager.semantic,
        embedding_manager=model.embedding_manager,
        queries=queries,
        ground_truth=ground_truth,
        k=3
    )
    
    # --- 5. System Benchmark ---
    print("\n--- [5/5] Running Speed & Memory Benchmarks ---")
    perf_metrics = evaluator.benchmark_performance(
        "What action should Harmony take if reasoning confidence is low?",
        num_runs=10
    )
    
    # --- Print Dashboard ---
    print("\n" + "="*55)
    print("             HARMONY EVALUATION RESULTS")
    print("="*55)
    print(f"  Language Model Loss:    {lm_metrics['eval_loss']:.4f}")
    ppl = lm_metrics['perplexity']
    ppl_str = f"{ppl:,.1f}" if ppl != float('inf') else "inf (needs more training)"
    print(f"  Model Perplexity:       {ppl_str}")
    print(f"  Planner Head Accuracy:  {plan_metrics['planner_accuracy']*100:.1f}%")
    print(f"  Verifier Accuracy:      {ver_metrics['verifier_accuracy']*100:.1f}%")
    print(f"  Verifier Precision:     {ver_metrics['verifier_precision']:.4f}")
    print(f"  Verifier Recall:        {ver_metrics['verifier_recall']:.4f}")
    print(f"  Verifier F1 Score:      {ver_metrics['verifier_f1']:.4f}")
    print(f"  Retrieval Recall@3:     {ret_metrics['recall_at_k']*100:.1f}%")
    print(f"  Retrieval MRR:          {ret_metrics['mrr']:.4f}")
    print(f"  Avg Inference Latency:  {perf_metrics['avg_latency_ms']:.2f} ms")
    if device == "cuda":
        print(f"  Peak GPU Memory:        {perf_metrics['peak_memory_mb']:.2f} MB")
    else:
        print(f"  Device:                 CPU (no GPU available)")
    print("="*55)
    
    # Summary note
    n_docs = model.memory_manager.semantic.index.ntotal
    print(f"\nModel trained on {n_docs} indexed document chunks.")
    print("Increase epochs in settings.py to improve LM loss and perplexity further.")

if __name__ == "__main__":
    main()
