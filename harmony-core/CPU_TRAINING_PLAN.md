# HARMONY Model CPU Training Plan
**Device:** AMD Ryzen 7 7730U, 16GB RAM, AMD Radeon Graphics (496MB integrated)
**Storage Available:** 171 GB free
**Training Mode:** CPU-only (integrated GPU not suitable for deep learning)
**Date:** June 6, 2026

---

## Critical Device Constraints Analysis

### Hardware Reality Check
- **CPU:** AMD Ryzen 7 7730U (8 cores, 2.00 GHz base, up to 4.5 GHz boost) - Good for CPU training
- **RAM:** 16GB (15.4GB usable) - Moderate constraint, limits batch size and model size
- **GPU:** AMD Radeon Graphics (496MB) - **Integrated graphics, NOT suitable for PyTorch training**
- **Storage:** 171GB free - Sufficient for medium-sized datasets

### What This Means
- **GPU training is NOT feasible** on this device (integrated GPU lacks VRAM and compute)
- **CPU-only training** is the only viable option
- **Model scaling must be conservative** to fit in 16GB RAM
- **Dataset size must be managed** to avoid memory overflow
- **Training will be slow** but possible with proper optimization

---

## Realistic Training Strategy

### Phase 1: Immediate Improvements (CPU-Optimized)
**Timeline:** 1-2 weeks
**Goal:** Maximize performance within CPU constraints

#### 1.1 Backbone Pretraining on Medium Dataset
**Dataset:** WikiText-103 (smaller subset)
- **Size:** ~100MB-500MB subset (not full 5GB)
- **Tokens:** ~10-50M tokens
- **Why:** Real language data, manageable size for CPU
- **Download:** HuggingFace `wikitext-103-v1`

**Model Configuration (CPU-Optimized):**
```
Hidden Size: 384 (up from 256, fits in RAM)
Layers: 6 (up from 4)
Attention Heads: 6
Vocabulary: 50,257
Max Sequence Length: 512
Parameters: ~30M (up from ~10M)
```

**Training Parameters:**
```
Batch Size: 4 (CPU constraint)
Gradient Accumulation: 4 (effective batch size = 16)
Epochs: 10-15
Learning Rate: 5e-4 (higher for faster convergence)
Optimizer: AdamW
Weight Decay: 1e-5
Warmup Steps: 1000
Total Training Time: ~24-48 hours
```

**Expected Impact:**
- Perplexity reduction: 1.9M → 500K-1M
- Better language understanding
- Improved downstream task performance

#### 1.2 Planner Training on Real Reasoning Data
**Dataset:** GSM8K (full dataset)
- **Size:** 7,500 math word problems
- **Format:** Question + step-by-step reasoning
- **Why:** Real planning/reasoning traces
- **Download:** HuggingFace `gsm8k`

**Planner Configuration:**
```
Training Samples: 7,500 (up from 6)
Action Types: 10-15 planning actions
Training Epochs: 15
Learning Rate: 1e-4
Batch Size: 16
```

**Expected Impact:**
- Planner accuracy: 33% → 60-70%
- Real reasoning capabilities
- Meaningful planning evaluation

#### 1.3 Verifier Training Expansion
**Dataset:** Combined verification datasets
- **GSM8K:** 7,500 samples (correct + corrupted)
- **StrategyQA:** 2,300 samples
- **Total:** ~10,000 verification examples

**Verifier Configuration:**
```
Training Samples: 10,000 (up from 600)
Epochs: 20
Learning Rate: 1e-4
Batch Size: 32
```

**Expected Impact:**
- Verifier F1: 64% → 75-80%
- Better fact verification
- More reliable confidence calibration

---

### Phase 2: Model Scaling (Conservative)
**Timeline:** 2-3 weeks
**Goal:** Scale model within RAM constraints

#### 2.1 Medium Model Architecture
**Configuration:**
```
Hidden Size: 512 (fits in 16GB RAM)
Layers: 8
Attention Heads: 8
Parameters: ~80M
Memory Requirement: ~8-10GB RAM
```

**Training Adjustments:**
```
Batch Size: 2 (reduced for larger model)
Gradient Accumulation: 8 (effective batch size = 16)
Mixed Precision: Not available on CPU
Checkpointing: Every epoch
Training Time: ~48-72 hours per stage
```

**Expected Impact:**
- Better representation learning
- Improved task performance
- Still feasible on CPU

#### 2.2 Dataset Expansion
**Backbone Dataset:** OpenWebText (subset)
- **Size:** ~1GB subset (not full 40GB)
- **Tokens:** ~100M tokens
- **Why:** Diverse web text, better generalization
- **Download:** HuggingFace `openwebtext`

**Expected Impact:**
- Better language modeling
- Improved generalization
- Lower perplexity

---

### Phase 3: Advanced Training (CPU-Optimized)
**Timeline:** 3-4 weeks
**Goal:** Advanced techniques for CPU efficiency

#### 3.1 Efficient Training Techniques
**Implement:**
- Gradient checkpointing (reduce memory by 50%)
- CPU-specific optimizations (MKL, oneDNN)
- Data loading optimizations
- Efficient attention mechanisms

#### 3.2 Long Context Training
**Experiment with:**
- Context lengths: 1024, 2048, 4096
- Rotary positional embeddings
- Memory-efficient attention

**Expected Impact:**
- Better long-context understanding
- Improved retrieval integration

#### 3.3 Memory System Enhancement
**Implement:**
- Importance scoring
- Memory promotion
- Memory pruning
- Episodic memory evaluation

**Expected Impact:**
- Better memory utilization
- Improved long-term reasoning

---

## Cloud GPU Option (Recommended for Major Scaling)

### Why Consider Cloud GPU
Your device constraints make significant scaling difficult. Cloud GPU enables:
- **10-50x faster training**
- **Larger models (768 hidden, 12+ layers)**
- **Full datasets (WikiText-103, OpenWebText, FineWeb)**
- **Mixed precision training**
- **Larger batch sizes**

### Recommended Cloud Options
**Budget-Friendly:**
- **Google Colab Pro:** $10/month (T4 GPU, 16GB VRAM)
- **Kaggle Kernels:** Free (P100/T4 GPU, limited hours)
- **RunPod:** $0.23/hour (RTX 4000 Ada, 20GB VRAM)

**Mid-Range:**
- **RunPod:** $0.40/hour (RTX 3090, 24GB VRAM)
- **Lambda Labs:** $0.60/hour (RTX 3090, 24GB VRAM)

**High-End:**
- **RunPod:** $0.80/hour (A100 40GB, 40GB VRAM)
- **Lambda Labs:** $1.10/hour (A100 40GB, 40GB VRAM)

### Cloud Training Plan
**If using cloud GPU (RTX 3090/A100):**

**Phase 1 (2-3 days):**
- Backbone: WikiText-103 full (5GB, 100M tokens)
- Model: 768 hidden, 12 layers (~200M params)
- Training: 50-100 epochs
- Expected: Perplexity < 100K

**Phase 2 (2-3 days):**
- Planner: GSM8K + MATH + StrategyQA (20k+ examples)
- Verifier: NLI + Fact verification datasets (50k+ examples)
- Training: 20-30 epochs each

**Phase 3 (3-5 days):**
- Long context: 4k, 8k, 16k benchmarks
- Memory system full implementation
- Scaling law experiments

**Total Cost:** ~$50-100 (depending on GPU choice)
**Total Time:** 7-11 days

---

## CPU Training Implementation Plan

### Immediate Actions (This Week)

#### Step 1: Dataset Preparation
```bash
# Create data directory
mkdir -p harmony-core/data/large_scale

# Download WikiText-103 subset
pip install datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')
# Save first 10% for CPU training
dataset = dataset.shard(num_shards=10, index=0)
dataset.save_to_disk('harmony-core/data/large_scale/wikitext_subset')
"

# Download GSM8K full
python -c "
from datasets import load_dataset
dataset = load_dataset('gsm8k', 'main')
dataset.save_to_disk('harmony-core/data/large_scale/gsm8k_full')
"

# Download StrategyQA
python -c "
from datasets import load_dataset
dataset = load_dataset('tasksource/strategyqa')
dataset.save_to_disk('harmony-core/data/large_scale/strategyqa')
"
```

#### Step 2: Model Configuration Update
Update `harmony-core/configs/model_config.yaml`:
```yaml
# CPU-optimized medium model
model:
  hidden_size: 384
  num_layers: 6
  num_attention_heads: 6
  intermediate_size: 1536
  max_position_embeddings: 512
  vocab_size: 50257
  dropout: 0.1

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-4
  num_epochs: 15
  warmup_steps: 1000
  weight_decay: 1e-5
  gradient_clip_norm: 1.0
```

#### Step 3: Training Script Optimization
Add CPU-specific optimizations:
```python
# Enable CPU optimizations
import torch
torch.set_num_threads(8)  # Use all CPU cores

# Enable MKL if available
try:
    import torch.backends.mkldnn as mkldnn
    mkldnn.enabled = True
except:
    pass

# Gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Pin memory for faster data loading
dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
```

### Training Schedule

**Week 1: Backbone Pretraining**
- Days 1-3: WikiText-103 subset training (384 hidden, 6 layers)
- Days 4-5: Evaluation and checkpointing
- Expected: 24-48 hours training time

**Week 2: Planner & Verifier Training**
- Days 1-2: GSM8K planner training (7,500 samples)
- Days 3-4: Verifier training (10,000 samples)
- Days 5: Joint fine-tuning
- Expected: 20-30 hours training time

**Week 3: Model Scaling**
- Days 1-3: OpenWebText subset training (512 hidden, 8 layers)
- Days 4-5: Evaluation and comparison
- Expected: 48-72 hours training time

**Week 4: Advanced Features**
- Days 1-2: Long context experiments
- Days 3-4: Memory system enhancements
- Days 5: Final evaluation
- Expected: 30-40 hours training time

---

## Resource Requirements

### Storage Requirements
- WikiText-103 subset: ~500MB
- GSM8K full: ~50MB
- StrategyQA: ~30MB
- OpenWebText subset: ~1GB
- Model checkpoints: ~2-5GB
- **Total: ~4-7GB** (well within 171GB available)

### RAM Requirements
- Training (384 hidden, 6 layers): ~6-8GB
- Training (512 hidden, 8 layers): ~8-10GB
- Inference: ~2-4GB
- **Peak: ~10GB** (fits within 16GB)

### CPU Utilization
- Training: 80-100% CPU usage
- Expected temperature increase: Monitor for thermal throttling
- Consider cooling solutions for long training runs

---

## Expected Results (CPU Training)

### Phase 1 Results
- **Perplexity:** 1.9M → 500K-1M
- **Planner Accuracy:** 33% → 60-70%
- **Verifier F1:** 64% → 75-80%
- **Training Time:** ~100-150 hours total

### Phase 2 Results
- **Perplexity:** 500K-1M → 200K-500K
- **Overall Task Performance:** 20-30% improvement
- **Training Time:** ~50-80 hours additional

### Phase 3 Results
- **Long Context:** Functional up to 4k tokens
- **Memory System:** Improved consolidation
- **Training Time:** ~30-40 hours additional

---

## Comparison: CPU vs Cloud GPU

| Metric | CPU Training | Cloud GPU (RTX 3090) |
|--------|--------------|---------------------|
| Training Speed | 1x baseline | 20-30x faster |
| Model Size | Max 512 hidden, 8 layers | 768 hidden, 12+ layers |
| Dataset Size | 100M tokens | 1B+ tokens |
| Training Time | 200-300 hours | 10-20 hours |
| Cost | Free | $50-100 |
| Perplexity Target | 200K-500K | < 50K |
| Final Quality | Good prototype | Production-ready |

---

## Recommendations

### Primary Recommendation: Hybrid Approach
1. **Start with CPU training** (Phase 1-2) to validate pipeline
2. **Use cloud GPU for final scaling** (Phase 3) to achieve production quality
3. **Total cost:** ~$20-50 for 2-3 days of GPU time
4. **Total time:** 4-5 weeks (CPU) + 2-3 days (GPU)

### Alternative: CPU-Only Approach
1. **Follow full CPU plan** (all phases)
2. **Accept limitations** on model size and dataset scale
3. **Total time:** 4-5 weeks
4. **Total cost:** Free
5. **Final quality:** Good prototype, not production-ready

### Alternative: Cloud-Only Approach
1. **Skip CPU training entirely**
2. **Use cloud GPU for all phases**
3. **Total time:** 1-2 weeks
4. **Total cost:** $50-100
5. **Final quality:** Production-ready

---

## Next Steps

### Immediate (Today)
1. Create data directory structure
2. Download WikiText-103 subset
3. Download GSM8K full dataset
4. Update model configuration for CPU-optimized medium model

### This Week
1. Implement CPU-specific training optimizations
2. Start Phase 1 backbone pretraining
3. Monitor training progress and resource usage

### Next Week
1. Complete Phase 1 training
2. Start Phase 2 planner/verifier training
3. Decide on cloud GPU strategy for Phase 3

---

## Monitoring and Evaluation

### Key Metrics to Track
- Training loss per epoch
- Validation perplexity
- CPU utilization and temperature
- RAM usage
- Training speed (samples/second)
- Disk I/O

### Evaluation Checkpoints
- After each training phase
- Compare against baseline (current model)
- Document improvements
- Adjust strategy based on results

---

## Conclusion

Your device can handle meaningful improvements to the HARMONY model through CPU-only training, but there are clear limitations:

**What's Possible on CPU:**
- 2-3x model scaling (256→512 hidden, 4→8 layers)
- 10x dataset expansion (local→WikiText subset)
- Real planner training (6→7,500 examples)
- Improved verifier (600→10,000 examples)
- Expected 2-3x overall performance improvement

**What Requires GPU:**
- Full-scale model (768+ hidden, 12+ layers)
- Full datasets (WikiText-103, OpenWebText, FineWeb)
- Production-level performance
- Reasonable training time

**Recommended Path:** Start with CPU training to validate improvements, then use cloud GPU for final scaling to achieve production-ready performance.
