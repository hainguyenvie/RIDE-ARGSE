# RIDE AR-GSE Pipeline - Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. Setup experts (download checkpoint manually first)
python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth

# 2. Train gating + optimize plugin + evaluate
python -m src.train.train_gating_only --mode selective
python run_improved_eg_outer.py
python -m src.train.eval_gse_plugin
```

## 📥 Download Pre-trained Models

**RIDE Standard (Recommended):**
- Link: https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view
- Performance: 48.6% overall, 25.7% few-shot

## 🎯 All Expert Setup Methods

| Command | Time | Performance | When to Use |
|---------|------|-------------|-------------|
| `--use-pretrained --pretrained-path <path>` | 15 min | ⭐⭐⭐⭐⭐ | **Recommended** |
| `--use-pretrained` (auto download) | 20 min | ⭐⭐⭐⭐⭐ | If gdown installed |
| `train_expert` (no args) | 6-8 hrs | ⭐⭐⭐⭐⭐ | Custom training |
| `manual_pretrained_setup` | 10 min | ⭐⭐⭐ | Quick fallback |

## 📋 Complete Pipeline Commands

### Option A: With Pre-trained Checkpoint (Fastest)
```bash
python -m src.train.train_expert --use-pretrained --pretrained-path ~/checkpoint.pth
python -m src.train.train_gating_only --mode selective  
python run_improved_eg_outer.py
python -m src.train.eval_gse_plugin
```

### Option B: Train from Scratch
```bash
python -m src.train.train_expert
python -m src.train.train_gating_only --mode selective
python run_improved_eg_outer.py
python -m src.train.eval_gse_plugin
```

### Option C: Interactive (Handles Errors)
```bash
python quick_start_ride.py
```

## 🔧 Troubleshooting One-Liners

```bash
# Check if experts are ready
ls outputs/logits/cifar100_lt_if100/ride_ce_expert/

# Re-download with gdown
pip install gdown
python -m src.train.train_expert --use-pretrained

# Use random init fallback
python -m src.utils.manual_pretrained_setup

# Load from checkpoint directly
python -m src.utils.load_pretrained /path/to/checkpoint.pth

# Check what's in a checkpoint
python -c "import torch; print(torch.load('checkpoint.pth', map_location='cpu').keys())"
```

## 📊 Expected Results

| Metric | Original | RIDE | Improvement |
|--------|----------|------|-------------|
| Overall Acc | ~39% | ~48% | +9% |
| Few-shot Acc | ~10% | ~25% | +15% |
| Training Time | 2-3 hrs | 15 min* | -85% |

*Using pre-trained weights

## 🎓 Key Concepts

- **RIDE Architecture**: Multi-expert model with shared early layers
- **3 Expert Variants**: CE, LogitAdjust, BalancedSoftmax (simulated from one model)
- **Diversity Loss**: KL divergence encourages expert specialization
- **Gating Network**: Learns to mix experts dynamically
- **Plugin Optimization**: Finds optimal α, μ for selective prediction

## 📁 Important Paths

```
checkpoints/experts/cifar100_lt_if100/     # Expert models
outputs/logits/cifar100_lt_if100/          # Exported logits
checkpoints/gating_pretrained/             # Gating network
checkpoints/argse_worst_eg_improved/       # Plugin results
results_worst_eg_improved/                 # Final metrics
```

## 🆘 Help Commands

```bash
# Show all options
python -m src.train.train_expert --help

# Direct checkpoint loading help
python -m src.utils.load_pretrained --help

# Interactive setup
python quick_start_ride.py
```

## ⚡ Pro Tips

1. **Always use `--pretrained-path`** for reliability
2. **Verify checkpoint download** (should be 2-10 MB, not KB)
3. **Use absolute paths** to avoid confusion
4. **Check logs carefully** - they show what's happening
5. **Keep the checkpoint** - reuse it anytime!
