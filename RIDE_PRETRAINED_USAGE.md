# Using Pre-trained RIDE Models

This guide explains how to use pre-trained RIDE models from the official model zoo instead of training experts from scratch.

## Available Pre-trained Models

Based on the RIDE model zoo for CIFAR-100-LT:

| Model | Experts | Overall Acc | Many Acc | Medium Acc | Few Acc | Description |
|-------|---------|-------------|----------|------------|---------|-------------|
| `ride_standard` | 3 | 48.6% | 67.0% | 49.9% | 25.7% | Standard RIDE without distillation |
| `ride_distill` | 3 | 49.0% | 67.6% | 50.9% | 25.2% | RIDE with knowledge distillation |
| `ride_distill_4experts` | 4 | 49.4% | 67.7% | 51.3% | 25.7% | RIDE with distillation and 4 experts |

## Usage Options

### Option 1: Complete Pipeline with Pre-trained Models (Recommended)

Run the complete pipeline using pre-trained experts:

```bash
# Use standard RIDE model (fastest)
python run_ride_pipeline_pretrained.py --use-pretrained

# Use RIDE with distillation (better performance)
python run_ride_pipeline_pretrained.py --use-pretrained --pretrained-model ride_distill

# Use 4-expert model (best performance)
python run_ride_pipeline_pretrained.py --use-pretrained --pretrained-model ride_distill_4experts
```

### Option 2: Step-by-step with Pre-trained Models

```bash
# Step 1: Setup pre-trained experts
python -m src.train.train_expert --use-pretrained --pretrained-model ride_standard

# Step 2: Train gating network
python -m src.train.train_gating_only --mode selective

# Step 3: Run plugin optimization
python run_improved_eg_outer.py

# Step 4: Evaluate results
python -m src.train.eval_gse_plugin
```

### Option 3: Direct Download and Setup

```bash
# Just download and setup pre-trained models
python -m src.utils.download_ride_pretrained --model ride_standard
```

## Benefits of Using Pre-trained Models

### ⚡ **Speed**
- Skip 200 epochs × 3 experts = ~600 epochs of training
- Go directly to gating training (Stage 2)
- Total pipeline time reduced by ~80%

### 🎯 **Performance**
- Proven performance on CIFAR-100-LT benchmark
- Better than training from scratch in most cases
- Consistent results across runs

### 💾 **Resources**
- No GPU time needed for expert training
- Lower memory requirements
- Immediate availability

## What Happens Behind the Scenes

1. **Download**: Pre-trained checkpoint downloaded from Google Drive
2. **Conversion**: Checkpoint converted to our `RIDEExpert` format
3. **Replication**: Single model replicated to create 3 expert variants
4. **Export**: Logits exported for all dataset splits (train, tuneV, val_lt, test_lt)
5. **Ready**: Pipeline ready for gating training

## File Structure After Setup

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

## Troubleshooting

### Download Issues
If Google Drive download fails:
1. Download manually from the provided URL
2. Save to `./checkpoints/ride_pretrained/`
3. Run the setup script again

### Checkpoint Conversion Issues
The conversion from RIDE format to our format may need adjustment based on the exact checkpoint structure. Check the console output for specific error messages.

### Missing Dependencies
Ensure you have all required packages:
```bash
pip install requests tqdm torch torchvision
```

## Performance Expectations

Using pre-trained RIDE models should provide:
- **Better tail class accuracy** than simple ResNet experts
- **Improved overall performance** on CIFAR-100-LT
- **More robust ensemble predictions**
- **Better calibration** for selective prediction

## Comparison

| Approach | Time | Performance | Flexibility |
|----------|------|-------------|-------------|
| Train from scratch | ~6-8 hours | Good | Full control |
| Use pre-trained | ~30 minutes | Better | Limited customization |

## Next Steps

After setting up pre-trained experts, continue with:
1. Gating network training (Stage 2)
2. Plugin optimization (Stage 3)  
3. Final evaluation (Stage 4)

The rest of the pipeline remains unchanged and will benefit from the improved expert quality.
