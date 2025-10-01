#!/usr/bin/env python3
"""
Complete RIDE-based AR-GSE Pipeline Runner
This script runs the full pipeline with RIDE-based experts for improved performance.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False

def main():
    """Run the complete RIDE-based AR-GSE pipeline"""
    print("🎯 RIDE-based AR-GSE Training Pipeline")
    print("=" * 60)
    print("This pipeline uses RIDE methodology for improved expert training:")
    print("✅ Multi-expert architecture with shared early layers")
    print("✅ Diversity-aware training with KL regularization")
    print("✅ Distribution-aware expert specialization")
    print("✅ Improved performance on long-tailed datasets")
    print("=" * 60)
    
    # Check if data splits exist
    splits_dir = Path("./data/cifar100_lt_if100_splits")
    if not splits_dir.exists():
        print(f"❌ Data splits directory not found: {splits_dir}")
        print("Please run data preparation first!")
        return False
    
    required_splits = ["train_indices.json", "tuneV_indices.json", "val_lt_indices.json", "test_lt_indices.json"]
    for split_file in required_splits:
        if not (splits_dir / split_file).exists():
            print(f"❌ Required split file not found: {split_file}")
            return False
    
    print("✅ All required data splits found")
    
    # Step 1: Train RIDE-based experts
    if not run_command(
        [sys.executable, "-m", "src.train.train_expert"],
        "Training RIDE-based Experts (Stage 1)"
    ):
        return False
    
    # Step 2: Train gating network in selective mode
    if not run_command(
        [sys.executable, "-m", "src.train.train_gating_only", "--mode", "selective"],
        "Training Selective Gating Network (Stage 2)"
    ):
        return False
    
    # Step 3: Run improved GSE plugin optimization
    if not run_command(
        [sys.executable, "run_improved_eg_outer.py"],
        "GSE Plugin Optimization with Improvements (Stage 3)"
    ):
        return False
    
    # Step 4: Evaluate final results
    if not run_command(
        [sys.executable, "-m", "src.train.eval_gse_plugin"],
        "Final Evaluation and Results (Stage 4)"
    ):
        return False
    
    print("\n" + "=" * 60)
    print("🎉 RIDE-based AR-GSE Pipeline Completed Successfully!")
    print("=" * 60)
    print("Results saved to:")
    print("📁 Expert checkpoints: ./checkpoints/experts/cifar100_lt_if100/")
    print("📁 Expert logits: ./outputs/logits/cifar100_lt_if100/")
    print("📁 Gating checkpoint: ./checkpoints/gating_pretrained/cifar100_lt_if100/")
    print("📁 Plugin checkpoint: ./checkpoints/argse_worst_eg_improved/cifar100_lt_if100/")
    print("📁 Final results: ./results_worst_eg_improved/cifar100_lt_if100/")
    print("=" * 60)
    print("\nKey improvements from RIDE methodology:")
    print("• Enhanced expert diversity through KL regularization")
    print("• Better distribution-aware specialization")
    print("• Improved performance on tail classes")
    print("• More robust ensemble predictions")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
