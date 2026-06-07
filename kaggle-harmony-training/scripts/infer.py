import torch
from harmony.models.harmony_model import HarmonyModel

def main():
    print("Loading HARMONY-Core v1 Phase B...")
    model = HarmonyModel()
    model.eval()
    
    text = "HARMONY-Core now has planner and verifier heads!"
    print(f"\nInput text: {text}")
    
    # Use standard pipeline
    with torch.no_grad():
        results = model.process_text(text)
        
        # Test retry loop
        print("\nTesting Confidence-based Retry Loop:")
        chunked_ids = results["chunked_ids"]
        retry_results = model.generate_with_retry(chunked_ids, max_retries=3, threshold=0.5)
        
    print(f"Action Taken: {retry_results['action']}")
    print(f"Confidence Score: {retry_results['confidence']:.4f}")
    print(f"Retries: {retry_results['retries']}")
    
    print("\nPhase B Inference pipeline executed successfully!")

if __name__ == "__main__":
    main()
