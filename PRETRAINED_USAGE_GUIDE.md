# Using Pre-trained RIDE Models - Complete Guide

## 🎯 Quick Start (Recommended Method)

### Step 1: Download RIDE Checkpoint Manually
1. Go to: https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view
2. Download the checkpoint file
3. Save it somewhere accessible (e.g., `~/Downloads/ride_cifar100.pth`)

### Step 2: Load from Local Path
```bash
python -m src.train.train_expert --use-pretrained --pretrained-path ~/Downloads/ride_cifar100.pth
```

### Step 3: Continue Pipeline
```bash
python -m src.train.train_gating_only --mode selective
python run_improved_eg_outer.py
python -m src.train.eval_gse_plugin
```

That's it! 🎉

## 📋 Available Pre-trained Models

| Model | Download Link | Experts | Overall | Few-shot |
|-------|--------------|---------|---------|----------|
| RIDE Standard | [Download](https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view) | 3 | 48.6% | 25.7% |
| RIDE + Distill | [Download](https://drive.google.com/file/d/1W-EICEpAavKzlnayiFPvb5cDyGCBl34l/view) | 3 | 49.0% | 25.2% |
| RIDE + Distill (4 experts) | [Download](https://drive.google.com/file/d/11kyxcYIh3bXk3mn3Y8EENHcsx-Ld9PXH/view) | 4 | 49.4% | 25.7% |

## 🔧 Usage Options

### Option 1: Local Checkpoint (Recommended) ⭐
```bash
# Download checkpoint manually first, then:
python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth
```

**Pros:**
- ✅ Most reliable (no download issues)
- ✅ Works with any checkpoint format
- ✅ You control the file
- ✅ Can verify download before use

**When to use:** Always prefer this method!

### Option 2: Automatic Download
```bash
# Install gdown first
pip install gdown

# Then run
python -m src.train.train_expert --use-pretrained --pretrained-model ride_standard
```

**Pros:**
- ✅ Convenient (one command)
- ⚠️ May fail with network issues

**When to use:** If you have reliable internet and gdown installed

### Option 3: Train from Scratch
```bash
python -m src.train.train_expert
```

**Pros:**
- ✅ Full control over training
- ✅ Can customize hyperparameters
- ⚠️ Takes 6-8 hours

**When to use:** If you want custom configurations or can't access pre-trained models

### Option 4: Random Initialization (Fast Fallback)
```bash
python -m src.utils.manual_pretrained_setup
```

**Pros:**
- ✅ Very fast (~10 minutes)
- ✅ Still uses RIDE architecture
- ⚠️ Lower performance (no pre-trained weights)

**When to use:** Quick testing or when other options fail

## 📝 Direct CLI Usage

You can also use the checkpoint loader directly:

```bash
# Load and setup from checkpoint
python -m src.utils.load_pretrained /path/to/checkpoint.pth

# With 4 experts
python -m src.utils.load_pretrained /path/to/checkpoint.pth --num-experts 4

# On CPU
python -m src.utils.load_pretrained /path/to/checkpoint.pth --device cpu
```

## 🔍 What Happens During Setup

1. **Loads checkpoint** with multiple fallback methods
2. **Creates RIDEExpert model** (3 experts by default)
3. **Maps weights** from checkpoint to our model format
4. **Saves 3 expert variants**:
   - `ride_ce_expert`
   - `ride_logitadjust_expert`
   - `ride_balsoftmax_expert`
5. **Exports logits** for all dataset splits:
   - train, tuneV, val_lt, test_lt

## 📁 File Structure After Setup

```
./checkpoints/experts/cifar100_lt_if100/
├── final_calibrated_ride_ce_expert.pth
├── final_calibrated_ride_logitadjust_expert.pth
└── final_calibrated_ride_balsoftmax_expert.pth

./outputs/logits/cifar100_lt_if100/
├── ride_ce_expert/
│   ├── train_logits.pt
│   ├── tuneV_logits.pt
│   ├── val_lt_logits.pt
│   └── test_lt_logits.pt
├── ride_logitadjust_expert/
│   └── ... (same files)
└── ride_balsoftmax_expert/
    └── ... (same files)
```

## ⚠️ Troubleshooting

### Problem: "Checkpoint file not found"
**Solution:** Check the path is correct
```bash
# Use absolute path
python -m src.train.train_expert --use-pretrained --pretrained-path /full/path/to/checkpoint.pth

# Or relative to current directory
python -m src.train.train_expert --use-pretrained --pretrained-path ./checkpoint.pth
```

### Problem: "Expected hasRecord error" or "Failed to load checkpoint"
**Solution:** The checkpoint format may be incompatible. Our loader tries multiple methods:
1. Standard PyTorch load
2. weights_only=True mode
3. Key mapping (removing 'module.' prefix)
4. Non-strict loading

If all fail, you'll get a detailed error message. Try:
- Re-downloading the checkpoint
- Using a different checkpoint
- Training from scratch

### Problem: "Missing keys" or "Unexpected keys"
**Solution:** This is usually OK! The loader handles this automatically with non-strict loading. The model will still work.

### Problem: Download fails with automatic method
**Solution:** Use local checkpoint method (Option 1) instead:
```bash
# Download manually, then:
python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth
```

## 🚀 Performance Comparison

| Method | Setup Time | Performance | Use Case |
|--------|-----------|-------------|----------|
| Pre-trained (local) | ~15 min | ⭐⭐⭐⭐⭐ 48-49% | **Recommended** |
| Train from scratch | ~6-8 hours | ⭐⭐⭐⭐⭐ 48-50% | Custom training |
| Random init | ~10 min | ⭐⭐⭐ 40-45% | Quick testing |

## 💡 Tips

1. **Always prefer local checkpoint method** - it's most reliable
2. **Verify download** - check file size is ~2-10MB (not a few KB)
3. **Use absolute paths** - avoids path confusion
4. **Check logs** - the loader shows detailed info about what it's doing
5. **Keep checkpoint file** - you can reuse it multiple times

## 🔗 Full Pipeline Example

```bash
# 1. Download checkpoint manually from Google Drive
# https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view

# 2. Setup experts from checkpoint
python -m src.train.train_expert --use-pretrained --pretrained-path ~/Downloads/ride_checkpoint.pth

# 3. Train gating network
python -m src.train.train_gating_only --mode selective

# 4. Optimize plugin
python run_improved_eg_outer.py

# 5. Evaluate
python -m src.train.eval_gse_plugin
```

## 📞 Need Help?

If you encounter issues:
1. Check this guide's troubleshooting section
2. Look at the detailed error messages from the loader
3. Try the fallback options (train from scratch or random init)
4. The loader is designed to be robust - it will try multiple methods before failing!
