#!/usr/bin/env python3
"""
AR-GSE Pipeline with Pre-trained RIDE Experts
This script allows using pre-trained RIDE models from the model zoo instead of training from scratch.
"""

import subprocess
import sys
import argparse
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
    """Run the AR-GSE pipeline with pre-trained or trained experts"""
    parser = argparse.ArgumentParser(description='AR-GSE Pipeline with RIDE Experts')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pre-trained RIDE models from model zoo')
    parser.add_argument('--pretrained-model', 
                       choices=['ride_standard', 'ride_distill', 'ride_distill_4experts'],
                       default='ride_standard',
                       help='Which pre-trained model to use')
    parser.add_argument('--skip-expert-training', action='store_true',
                       help='Skip expert training (assumes experts already exist)')
    
    args = parser.parse_args()
    
    print("🎯 AR-GSE Training Pipeline with RIDE Experts")
    print("=" * 60)
    
    if args.use_pretrained:
        print("🔄 Mode: Using Pre-trained RIDE Models")
        print(f"📦 Model: {args.pretrained_model}")
        print("✅ Benefits: Skip training time, proven performance")
    elif args.skip_expert_training:
        print("🔄 Mode: Skip Expert Training (use existing)")
    else:
        print("🔄 Mode: Train RIDE Experts from Scratch")
        print("✅ Benefits: Full control, custom configurations")
    
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
    
    # Step 1: Setup experts (pre-trained or train from scratch)
    if args.use_pretrained:
        if not run_command(
            [sys.executable, "-m", "src.utils.download_ride_pretrained", 
             "--model", args.pretrained_model],
            "Setting up Pre-trained RIDE Experts (Stage 1)"
        ):
            return False
    elif not args.skip_expert_training:
        if not run_command(
            [sys.executable, "-m", "src.train.train_expert"],
            "Training RIDE-based Experts (Stage 1)"
        ):
            return False
    else:
        print("\n⏭️ Skipping expert training (using existing experts)")
        
        # Check if expert logits exist
        logits_dir = Path("./outputs/logits/cifar100_lt_if100")
        expert_names = ['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']
        
        missing_experts = []
        for expert_name in expert_names:
            expert_logits_dir = logits_dir / expert_name
            if not expert_logits_dir.exists() or not (expert_logits_dir / "tuneV_logits.pt").exists():
                missing_experts.append(expert_name)
        
        if missing_experts:
            print(f"❌ Missing expert logits for: {missing_experts}")
            print("Please run expert training first or use --use-pretrained")
            return False
        
        print("✅ Found existing expert logits")
    
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
    print("🎉 AR-GSE Pipeline Completed Successfully!")
    print("=" * 60)
    
    if args.use_pretrained:
        print(f"Used pre-trained model: {args.pretrained_model}")
        print("Benefits achieved:")
        print("  ⚡ Significantly faster pipeline execution")
        print("  🎯 Proven expert performance on CIFAR-100-LT")
        print("  📊 Expected performance improvements")
    
    print("\nResults saved to:")
    print("📁 Expert checkpoints: ./checkpoints/experts/cifar100_lt_if100/")
    print("📁 Expert logits: ./outputs/logits/cifar100_lt_if100/")
    print("📁 Gating checkpoint: ./checkpoints/gating_pretrained/cifar100_lt_if100/")
    print("📁 Plugin checkpoint: ./checkpoints/argse_worst_eg_improved/cifar100_lt_if100/")
    print("📁 Final results: ./results_worst_eg_improved/cifar100_lt_if100/")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
