# HARMONY Model - Google Colab Guide

**Beginner-friendly guide for testing and training the HARMONY cognitive architecture on Google Colab**

---

## Table of Contents

1. [What is Google Colab?](#what-is-google-colab)
2. [Getting Started](#getting-started)
3. [Quick Testing Mode](#quick-testing-mode)
4. [Full Training Mode](#full-training-mode)
5. [Managing Sessions](#managing-sessions)
6. [Saving and Loading Models](#saving-and-loading-models)
7. [Troubleshooting](#troubleshooting)
8. [Tips for Success](#tips-for-success)

---

## What is Google Colab?

Google Colab (Colaboratory) is a free cloud service that provides:
- **Free GPU access** (T4 GPU on free tier)
- **Jupyter notebook environment** in your browser
- **No setup required** - everything runs in the cloud
- **Google Drive integration** for saving files

### Free Colab Limitations

- **Session timeout**: 12 hours maximum per session
- **GPU availability**: Not guaranteed (may get CPU only)
- **Disk space**: ~30GB temporary storage
- **GPU type**: T4 (16GB VRAM) on free tier

**Don't worry!** Our notebook is designed to work within these limits.

---

## Getting Started

### Step 1: Open the Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "File" → "Open notebook"
3. Upload `harmony_colab.ipynb` from your computer
4. Or open it from GitHub if you've uploaded it there

### Step 2: Enable GPU

1. Click "Runtime" in the top menu
2. Click "Change runtime type"
3. Select "GPU" under "Hardware accelerator"
4. Click "Save"

**Note**: If GPU is not available, the notebook will still work on CPU (just slower).

### Step 3: Run Setup Section

The notebook has a **Setup & Installation** section at the top. Run all cells in this section first:

1. Click the first cell in the setup section
2. Click the "Play" button (▶) or press `Shift + Enter`
3. Wait for the cell to finish (look for the checkmark ✓)
4. Continue with the next cell

**What the setup does:**
- Checks if GPU is available
- Clones the HARMONY repository
- Installs all required Python packages
- Mounts Google Drive for saving models
- Verifies everything is ready

**Time**: 5-10 minutes (depends on internet speed)

### Step 4: Choose Your Mode

After setup, you have two options:

- **🚀 Quick Testing**: Test the model with pre-trained weights (5-10 minutes)
- **🎯 Full Training**: Train the model from scratch (2-3 hours)

Skip to the section you want to use.

---

## Quick Testing Mode

**Best for**: Understanding the model, running demos, quick experiments

### Prerequisites

- A pre-trained model file (`.pt`) in your Google Drive
- OR you can test with untrained model (random weights)

### How to Use

1. **Skip to the "🚀 Quick Testing" section** in the notebook
2. Run the cells one by one from top to bottom

### What Happens

1. **Initialize Colab Manager**: Sets up Google Drive and monitoring
2. **Download Model**: Attempts to download pre-trained model from Drive
3. **Load Model**: Loads HARMONY model (with or without pre-trained weights)
4. **Test Inference**: Runs the model on sample queries
5. **Test Planner**: Tests the planning component
6. **Test Memory**: Tests the retrieval memory system

### Expected Output

You'll see responses like:
```
Query: What is the capital of France?
Response: Paris is the capital of France...
```

### If No Pre-trained Model

The notebook will warn you:
```
⚠ No pre-trained model found in Drive
```

You can:
- Upload a model to Google Drive and retry
- Skip to Full Training to train from scratch
- Continue with untrained model (for testing architecture only)

### Time Required

- **With pre-trained model**: 5-10 minutes
- **Without pre-trained model**: 5-10 minutes (but responses will be random)

---

## Full Training Mode

**Best for**: Training custom models, research, production-ready models

### Prerequisites

- None! The notebook downloads everything needed
- Google Drive mounted (for saving checkpoints)

### How to Use

1. **Skip to the "🎯 Full Training Pipeline" section**
2. Run cells one by one from top to bottom
3. Each cell corresponds to a training stage

### Training Stages

The training follows a 5-stage curriculum:

#### Stage 1: Backbone Pretraining (30-60 minutes)

**Purpose**: Teach the model to predict next tokens in text

**Dataset**: WikiText-103 (subset, ~100K tokens)

**What happens**:
- Downloads WikiText-103 dataset
- Tokenizes the text
- Trains the language model
- Saves checkpoints every 2 epochs

**Expected output**:
```
Epoch 1/8
  Step 0: Loss = 10.2341
  Step 50: Loss = 8.1234
  Average loss: 7.8901
✓ Checkpoint saved: stage1_epoch2
```

#### Stage 2: Retrieval Memory Setup (5-10 minutes)

**Purpose**: Build a vector database for knowledge retrieval

**What happens**:
- Creates a knowledge ingestion pipeline
- Adds WikiText documents to memory
- Saves the vector index

**Expected output**:
```
Adding 100 documents to memory...
✓ Memory seeded with 100 vectors
✓ Vector memory saved to /content/checkpoints/vector_memory
```

#### Stage 3: Planner Training (20-40 minutes)

**Purpose**: Train the planner to generate action sequences

**Dataset**: GSM8K math problems (5,000 samples)

**What happens**:
- Prepares planner training data
- Trains the planner component
- Tracks accuracy improvement

**Expected output**:
```
Epoch 1/12
  Step 0: Loss = 2.3456, Acc = 10.0%
  Step 20: Loss = 1.8901, Acc = 45.2%
  Average loss: 1.7654, Accuracy: 52.3%
```

#### Stage 4: Verifier Training (20-40 minutes)

**Purpose**: Train the verifier to distinguish correct from incorrect answers

**Dataset**: GSM8K (5,000 correct + 5,000 incorrect examples)

**What happens**:
- Creates verification dataset
- Trains the verifier component
- Tracks accuracy

**Expected output**:
```
Epoch 1/15
  Step 0: Loss = 0.6931, Acc = 50.0%
  Step 20: Loss = 0.4567, Acc = 78.5%
  Average loss: 0.4234, Accuracy: 82.1%
```

#### Stage 5: Joint Fine-Tuning (15-30 minutes)

**Purpose**: Fine-tune all components together

**What happens**:
- Trains all components simultaneously
- Balances losses from different tasks
- Produces final model

**Expected output**:
```
Epoch 1/8
  Step 0: Loss = 5.6789 (LM: 4.1234, Planner: 1.2345, Verifier: 0.3210)
  Average loss: 4.5678
```

### Total Time

- **Minimum**: 1.5-2 hours (if everything runs smoothly)
- **Typical**: 2-3 hours (includes downloads, checkpoints)
- **Can be split**: Use checkpoints to resume later

### Expected Results

After training on Colab T4 GPU:

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Perplexity | ~1.9M | ~100K-500K |
| Planner Accuracy | ~33% | ~60-70% |
| Verifier F1 | ~64% | ~75-80% |
| Model Size | 256 hidden, 4 layers | 384 hidden, 6 layers |

---

## Managing Sessions

### Session Timeout

Colab sessions timeout after 12 hours. Here's how to handle it:

#### Before Timeout

1. **Checkpoints are saved automatically** after each stage
2. **Models are saved to Google Drive** automatically
3. **You can resume** from the last checkpoint

#### After Timeout

1. Re-open the notebook
2. Run the setup section again
3. Use the checkpoint loading code (see below)

### Resuming from Checkpoint

If your session times out, you can resume:

```python
# Load the last checkpoint
checkpoint_path = "/content/drive/MyDrive/HARMONY/stage3_final.pt"
checkpoint = colab_manager.load_checkpoint(checkpoint_path, model, optimizer)

# Continue training from the saved epoch
start_epoch = checkpoint['epoch'] + 1
```

### Splitting Training Across Sessions

You can train different stages in different sessions:

**Session 1**: Stages 1-2 (Backbone + Memory)
**Session 2**: Stages 3-4 (Planner + Verifier)
**Session 3**: Stage 5 (Joint Fine-Tuning)

Just run the setup section at the start of each session.

---

## Saving and Loading Models

### Automatic Saving

The notebook automatically saves:
- **Checkpoints**: After each training stage
- **Final model**: At the end of training
- **To both locations**: Local `/content/` and Google Drive

### Manual Saving

To manually save the current model:

```python
colab_manager.save_checkpoint(model, "manual_checkpoint", optimizer, epoch)
```

### Loading a Model

To load a saved model:

```python
# From Google Drive
colab_manager.download_from_drive("harmony_trained.pt", "/content")

# Load the model
checkpoint = torch.load("/content/harmony_trained.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
```

### Downloading to Your Computer

1. Open Google Drive in your browser
2. Navigate to `My Drive/HARMONY/`
3. Download the `.pt` files you want
4. Use them locally with the HARMONY codebase

### Uploading to Colab

To use your own pre-trained model:

1. Upload the `.pt` file to Google Drive
2. Place it in `My Drive/HARMONY/`
3. The notebook will automatically detect it

---

## Troubleshooting

### Problem: "No GPU Available"

**Cause**: GPU resources are busy or unavailable

**Solutions**:
1. Wait a few minutes and try again
2. Continue on CPU (slower but works)
3. Try at a different time of day (less traffic)

### Problem: "Out of Memory" (OOM)

**Cause**: Model or batch size too large for GPU

**Solutions**:
1. Reduce `batch_size` in `config_colab.yaml`
2. Reduce `hidden_size` from 384 to 256
3. Reduce `num_layers` from 6 to 4
4. Restart the kernel and try again

### Problem: "Session Timeout"

**Cause**: Colab 12-hour limit reached

**Solutions**:
1. Checkpoints are saved - you can resume
2. Re-open the notebook and run setup
3. Load the last checkpoint
4. Continue training

### Problem: "Import Error"

**Cause**: Dependencies not installed

**Solutions**:
1. Make sure you ran the setup section
2. Run `!pip install -e .` again
3. Restart the kernel
4. Run setup again

### Problem: "Slow Training"

**Cause**: Various factors (dataset size, GPU, etc.)

**Solutions**:
1. Check you're using GPU (not CPU)
2. Reduce dataset size in config
3. Use mixed precision (enabled by default)
4. Reduce number of epochs

### Problem: "Google Drive Mount Failed"

**Cause**: Permission issues or network problems

**Solutions**:
1. Click the link in the authorization popup
2. Sign in to your Google account
3. Copy the authorization code
4. Paste it in the notebook
5. Try mounting again

### Problem: "Dataset Download Failed"

**Cause**: Network issues or HuggingFace down

**Solutions**:
1. Check your internet connection
2. Try again in a few minutes
3. Use a smaller dataset subset
4. Download manually and upload to Drive

---

## Tips for Success

### Before You Start

1. **Use Google Chrome** for best Colab experience
2. **Enable GPU** before running any cells
3. **Mount Google Drive** to save your work
4. **Clear browser cache** if you have issues

### During Training

1. **Monitor GPU memory** with the memory status cells
2. **Save frequently** - checkpoints are automatic
3. **Don't close the tab** while training is running
4. **Keep the tab active** to prevent timeout

### For Better Performance

1. **Use T4 GPU** (most common on free Colab)
2. **Enable mixed precision** (FP16) - enabled by default
3. **Use gradient accumulation** for larger effective batch size
4. **Reduce dataset size** if training is too slow

### For Longer Training

1. **Split across sessions** using checkpoints
2. **Save to Google Drive** for persistence
3. **Monitor session time** (12-hour limit)
4. **Use Colab Pro** for longer sessions (optional)

### For Beginners

1. **Start with Quick Testing** to understand the model
2. **Read the comments** in each cell
3. **Run cells sequentially** - don't skip
4. **Check outputs** after each cell
5. **Ask for help** if you get stuck

### For Advanced Users

1. **Modify config_colab.yaml** for custom settings
2. **Add your own datasets** for training
3. **Experiment with hyperparameters**
4. **Use custom checkpoints** for transfer learning

---

## Advanced Usage

### Using Your Own Data

To train with your own dataset:

1. Upload your data to Google Drive
2. Modify the dataset loading code in the notebook
3. Adjust the configuration in `config_colab.yaml`
4. Run training as usual

### Transfer Learning

To fine-tune a pre-trained model:

1. Load a pre-trained checkpoint
2. Freeze some layers if desired
3. Train on your new data
4. Save the fine-tuned model

### Hyperparameter Tuning

To experiment with settings:

1. Edit `config_colab.yaml`
2. Change values like:
   - `learning_rate`: Try 1e-4, 5e-4, 1e-3
   - `batch_size`: Try 4, 8, 16
   - `hidden_size`: Try 256, 384, 512
3. Run training and compare results

---

## Frequently Asked Questions

### Q: Is Colab really free?

**A**: Yes! The free tier provides:
- T4 GPU (when available)
- 12-hour sessions
- 30GB disk space
- No credit card required

### Q: Can I use this on my local machine?

**A**: Yes! The notebook can be adapted for local use:
- Install dependencies locally
- Remove Google Drive mounting
- Use local paths instead of `/content/`
- Use your local GPU if available

### Q: How do I get better GPUs?

**A**: Options for better GPUs:
- **Colab Pro** ($10/month): T4/V100, longer sessions
- **Colab Pro+** ($50/month): A100, best performance
- **Kaggle Kernels**: Free P100/T4 GPUs
- **RunPod/Lambda**: Paid cloud GPU services

### Q: Can I train on CPU only?

**A**: Yes, but it's much slower:
- Training time: 10-20x slower
- May need to reduce model size
- Still works for testing/small experiments

### Q: What if I lose my trained model?

**A**: As long as you saved to Google Drive:
- Models persist in Drive
- Download from Drive anytime
- Re-upload to Colab if needed

### Q: Can I share my trained model?

**A**: Yes! Options:
- Share the `.pt` file directly
- Upload to HuggingFace Hub
- Create a GitHub release
- Share via Google Drive link

---

## Getting Help

If you encounter issues:

1. **Check the Troubleshooting section** above
2. **Read cell comments** in the notebook
3. **Check GPU status** with `!nvidia-smi`
4. **Restart kernel** if needed
5. **Check HARMONY repository** for documentation

For more help:
- HARMONY GitHub repository
- Google Colab documentation
- PyTorch documentation

---

## Summary

**Quick Start Checklist**:

- [ ] Open notebook in Colab
- [ ] Enable GPU in runtime settings
- [ ] Run setup section
- [ ] Choose mode (Testing or Training)
- [ ] Run cells sequentially
- [ ] Monitor progress
- [ ] Save models to Drive
- [ ] Download results when done

**Key Points**:

- **Free Colab** is sufficient for training
- **Checkpoints** save your progress
- **Google Drive** persists your models
- **12-hour limit** - use checkpoints to resume
- **Both modes** work for beginners

**Happy Training! 🚀**
