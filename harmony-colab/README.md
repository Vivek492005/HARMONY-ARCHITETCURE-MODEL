# HARMONY Model - Google Colab Setup

Complete Google Colab materials for testing and training the HARMONY cognitive architecture model on free Colab GPUs.

## 📁 Files Included

- **`harmony_colab.ipynb`** - Main Colab notebook with testing and training modes
- **`COLAB_GUIDE.md`** - Comprehensive beginner-friendly guide
- **`setup_colab.py`** - Dependency installation script
- **`colab_utils.py`** - Utility functions for Drive integration and checkpointing
- **`config_colab.yaml`** - Colab-optimized configuration settings

## 🚀 Quick Start

### Option 1: Quick Testing (5-10 minutes)
Test the model with pre-trained weights or run inference examples.

### Option 2: Full Training (2-3 hours)
Train the model from scratch with all 5 stages on free Colab T4 GPU.

## 📖 Detailed Instructions

See **[COLAB_GUIDE.md](COLAB_GUIDE.md)** for:
- Step-by-step setup instructions
- How to use each mode
- Managing sessions and timeouts
- Saving and loading models
- Troubleshooting common issues
- Tips for success

## 🎯 Features

- **Two Modes**: Quick testing and full training in one notebook
- **Free Colab Optimized**: Works within 12-hour session limits
- **Google Drive Integration**: Automatic model saving and loading
- **Checkpoint System**: Resume training across sessions
- **Beginner-Friendly**: Extensive comments and explanations
- **Memory Monitoring**: GPU memory tracking and OOM prevention

## 📋 Requirements

- Google account (for Colab and Drive)
- Internet connection
- No local GPU required (runs in cloud)

## 🎓 Training Stages

1. **Stage 1**: Backbone Pretraining (WikiText-103)
2. **Stage 2**: Retrieval Memory Setup
3. **Stage 3**: Planner Training (GSM8K)
4. **Stage 4**: Verifier Training (GSM8K)
5. **Stage 5**: Joint Fine-Tuning

## 💡 Usage

1. Open `harmony_colab.ipynb` in Google Colab
2. Enable GPU in runtime settings
3. Run the setup section
4. Choose your mode (Testing or Training)
5. Follow the guide for detailed instructions

## 🔧 Configuration

Edit `config_colab.yaml` to customize:
- Model architecture (hidden size, layers)
- Training parameters (batch size, learning rate)
- Dataset sizes
- Checkpointing settings

## 📊 Expected Results

After training on free Colab T4 GPU:
- **Perplexity**: ~100K-500K (vs 1.9M baseline)
- **Planner Accuracy**: ~60-70% (vs 33% baseline)
- **Verifier F1**: ~75-80% (vs 64% baseline)
- **Model Size**: 384 hidden, 6 layers

## 🆘 Troubleshooting

See the [Troubleshooting section](COLAB_GUIDE.md#troubleshooting) in the guide for common issues and solutions.

## 📝 License

Part of the HARMONY project. See main repository for license details.

## 🤝 Support

For issues or questions:
- Check [COLAB_GUIDE.md](COLAB_GUIDE.md)
- Review the HARMONY repository documentation
- Monitor GPU usage with `!nvidia-smi` in Colab
