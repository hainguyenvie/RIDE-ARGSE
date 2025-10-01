# Correct Pipeline Commands for RIDE AR-GSE

## ✅ Complete Pipeline with Pre-trained RIDE Checkpoint

### Prerequisites
1. Download RIDE checkpoint from Google Drive
2. Have data splits prepared in `./data/cifar100_lt_if100_splits/`

### Step-by-Step Commands

#### Step 1: Setup Experts from Pre-trained Checkpoint
```bash
python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth
```

**What this does:**
- Loads RIDE checkpoint (handles `backbone.` prefix automatically)
- Filters out Expert Assignment (EA) module keys
- Creates 3 expert variants with RIDE names:
  - `ride_ce_expert`
  - `ride_logitadjust_expert` 
  - `ride_balsoftmax_expert`
- Exports logits for all splits to `outputs/logits/cifar100_lt_if100/`

**Expected output structure:**
```
outputs/logits/cifar100_lt_if100/
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

#### Step 2: Train Selective Gating Network
```bash
python -m src.train.train_gating_only --mode selective
```

**What this does:**
- Loads logits from Step 1
- Uses expert names: `['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']`
- Trains gating network with selective training (Pinball loss)
- Saves to `checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt`

#### Step 3: Run Plugin Optimization
```bash
python run_improved_eg_outer.py
```

**What this does:**
- Loads logits from Step 1 (uses correct RIDE expert names)
- Loads gating from Step 2
- Optimizes α, μ, thresholds via plugin algorithm
- Saves to `checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt`

#### Step 4: Evaluate Results
```bash
python -m src.train.eval_gse_plugin
```

**What this does:**
- Loads plugin results from Step 3
- Loads test logits from Step 1 (uses correct RIDE expert names)
- Evaluates on test set
- Generates RC curves and metrics
- Saves to `results_worst_eg_improved/cifar100_lt_if100/`

## 🔧 Fixed Issues

### Issue 1: Expert Names Mismatch ✅
**Old names:** `ce_baseline`, `logitadjust_baseline`, `balsoftmax_baseline`
**New names:** `ride_ce_expert`, `ride_logitadjust_expert`, `ride_balsoftmax_expert`

**Files updated:**
- `src/train/train_expert.py` - Creates RIDE experts with new names
- `src/train/train_gating_only.py` - Uses new expert names
- `src/train/gse_balanced_plugin.py` - Uses new expert names ✅
- `src/train/eval_gse_plugin.py` - Uses new expert names

### Issue 2: Checkpoint Loading ✅
**Problems:**
- `backbone.` prefix in checkpoint keys
- Expert Assignment (EA) module keys not needed

**Solutions:**
- Automatic prefix removal in loader
- Filter out EA keys (`expert_help`, `confidence`, `routing`)
- Multiple fallback loading strategies

### Issue 3: Lambda Function Serialization ✅
**Problem:** Raw lambda in BasicBlock couldn't be pickled

**Solution:** Wrapped lambda in `LambdaLayer` nn.Module class

## 📝 Complete Example (Kaggle/Colab)

```bash
# Download RIDE checkpoint (manual)
# Save to: /kaggle/working/RIDE-ARGSE/cifar_standard_055148/checkpoint-epoch5.pth

# Step 1: Setup experts
!python -m src.train.train_expert --use-pretrained \
  --pretrained-path /kaggle/working/RIDE-ARGSE/cifar_standard_055148/checkpoint-epoch5.pth

# Step 2: Train gating
!python -m src.train.train_gating_only --mode selective

# Step 3: Optimize plugin
!python run_improved_eg_outer.py

# Step 4: Evaluate
!python -m src.train.eval_gse_plugin
```

## ⚠️ Common Errors and Solutions

### Error: "Missing logits file: ce_baseline/tuneV_logits.pt"
**Cause:** Old expert names in configuration
**Solution:** Already fixed! Files now use correct RIDE expert names.

### Error: "Missing keys / Unexpected keys" when loading checkpoint
**Cause:** Key prefix mismatch or EA module keys
**Solution:** Already fixed! Loader automatically handles this.

### Error: "lambda is not a Module subclass"
**Cause:** Raw lambda function in BasicBlock
**Solution:** Already fixed! Now uses LambdaLayer wrapper.

## 🎯 Verification Steps

After Step 1, verify logits exist:
```bash
ls outputs/logits/cifar100_lt_if100/
# Should show: ride_ce_expert  ride_logitadjust_expert  ride_balsoftmax_expert

ls outputs/logits/cifar100_lt_if100/ride_ce_expert/
# Should show: train_logits.pt  tuneV_logits.pt  val_lt_logits.pt  test_lt_logits.pt
```

After Step 2, verify gating checkpoint:
```bash
ls checkpoints/gating_pretrained/cifar100_lt_if100/
# Should show: gating_selective.ckpt  selective_training_logs.json
```

After Step 3, verify plugin checkpoint:
```bash
ls checkpoints/argse_worst_eg_improved/cifar100_lt_if100/
# Should show: gse_balanced_plugin.ckpt
```

After Step 4, verify results:
```bash
ls results_worst_eg_improved/cifar100_lt_if100/
# Should show: metrics.json  rc_curve.csv  rc_curve_comparison.png  etc.
```

## 🚀 Quick Troubleshooting

If Step 3 or 4 fails with missing logits:
```bash
# Check if expert logits exist
ls outputs/logits/cifar100_lt_if100/

# If only old expert names exist, re-run Step 1
python -m src.train.train_expert --use-pretrained \
  --pretrained-path /path/to/checkpoint.pth
```

If checkpoint loading fails:
```bash
# The loader will show detailed debugging info
# Look for which "Attempt" succeeded
# Expected: "Attempt 2: Removing 'backbone.' prefix... ✅"
```

## 📊 Expected Performance

With pre-trained RIDE checkpoint:
- Overall accuracy: ~48-49%
- Few-shot accuracy: ~25-26%
- Much better than baseline CE (~39% overall, ~10% few-shot)

Setup time: ~15-20 minutes (vs 6-8 hours training from scratch)
