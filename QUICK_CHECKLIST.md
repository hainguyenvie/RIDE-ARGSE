# Quick Checklist - RIDE AR-GSE Pipeline

## ✅ Pre-flight Checklist

- [ ] RIDE checkpoint downloaded
- [ ] Data splits exist in `./data/cifar100_lt_if100_splits/`
- [ ] Python dependencies installed (`pip install -r requirements.txt`)

## 🚀 Pipeline Execution

### ☑️ Step 1: Load Pre-trained RIDE
```bash
python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth
```
- [ ] No errors during loading
- [ ] Checkpoint loads successfully (Attempt 2 succeeds)
- [ ] EA keys filtered out (6 keys)
- [ ] 3 experts created
- [ ] Logits exported for all splits

**Verify:**
```bash
ls outputs/logits/cifar100_lt_if100/
# Should show: ride_ce_expert  ride_logitadjust_expert  ride_balsoftmax_expert
```

---

### ☑️ Step 2: Train Selective Gating
```bash
python -m src.train.train_gating_only --mode selective
```
- [ ] S1/S2 loaders created
- [ ] Temperatures fitted
- [ ] Training completes
- [ ] Checkpoint saved

**Verify:**
```bash
ls checkpoints/gating_pretrained/cifar100_lt_if100/
# Should show: gating_selective.ckpt
```

---

### ☑️ Step 3: Plugin Optimization
```bash
python run_improved_eg_outer.py
```
- [ ] Logits load correctly (RIDE names)
- [ ] No "ce_baseline not found" errors
- [ ] Optimization completes
- [ ] Checkpoint saved

**Verify:**
```bash
ls checkpoints/argse_worst_eg_improved/cifar100_lt_if100/
# Should show: gse_balanced_plugin.ckpt
```

---

### ☑️ Step 4: Evaluation
```bash
python -m src.train.eval_gse_plugin
```
- [ ] Plugin checkpoint loads
- [ ] Test logits load (RIDE names)
- [ ] Metrics computed
- [ ] Plots generated

**Verify:**
```bash
ls results_worst_eg_improved/cifar100_lt_if100/
# Should show: metrics.json, rc_curve.csv, *.png
```

---

## 🎯 Quick Verification

Run at any time:
```bash
python verify_setup.py
```

## ⚠️ Common Issues (All Fixed!)

| Issue | Status | Solution |
|-------|--------|----------|
| Lambda serialization error | ✅ Fixed | LambdaLayer wrapper added |
| backbone. prefix mismatch | ✅ Fixed | Auto prefix removal |
| ce_baseline not found | ✅ Fixed | Updated to RIDE names |
| EA keys unexpected | ✅ Fixed | Auto filtering |

## 📊 Expected Output

### Performance
- Overall: ~48-49%
- Few-shot: ~25-26%
- Improvement over baseline: +9% overall, +15% few-shot

### File Sizes (Approximate)
- Checkpoint: 3-10 MB
- Logits per split: 1-10 MB
- Results: < 1 MB

## 🔧 If Something Fails

1. **Check with verification script:**
   ```bash
   python verify_setup.py --step X
   ```

2. **Read error messages carefully** - they're detailed

3. **Check file existence:**
   ```bash
   ls outputs/logits/cifar100_lt_if100/
   ls checkpoints/gating_pretrained/cifar100_lt_if100/
   ```

4. **Re-run failed step** - often fixes transient issues

5. **Check FINAL_FIXES_SUMMARY.md** for detailed explanations

## 🎉 Success Indicators

### After Step 1:
```
✅ Weights loaded successfully (backbone. removed)
✅ Successfully created 3 experts!
```

### After Step 2:
```
✅ Selective Pinball training complete.
```

### After Step 3:
```
✅ Saved optimal parameters to ...gse_balanced_plugin.ckpt
```

### After Step 4:
```
💾 Saved metrics to ...metrics.json
📊 Saved RC curve plots
```

## 📚 Documentation

- **FINAL_FIXES_SUMMARY.md** - What was fixed and why
- **CORRECT_PIPELINE_COMMANDS.md** - Detailed step-by-step guide
- **PRETRAINED_USAGE_GUIDE.md** - Pre-trained model guide
- **QUICK_REFERENCE.md** - Command reference

## ✨ You're All Set!

All issues have been fixed. The pipeline should run smoothly from start to finish. Good luck! 🚀
