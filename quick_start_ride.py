#!/usr/bin/env python3
"""
Quick Start Script for RIDE-based AR-GSE Pipeline
Handles pre-trained model issues gracefully with multiple fallback options.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully!")
            return True
        else:
            print(f"\n⚠️ {description} completed with warnings (return code {result.returncode})")
            return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return False

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check data splits
    splits_dir = Path("./data/cifar100_lt_if100_splits")
    if not splits_dir.exists():
        print(f"❌ Data splits directory not found: {splits_dir}")
        return False
    
    required_splits = ["train_indices.json", "tuneV_indices.json", "val_lt_indices.json", "test_lt_indices.json"]
    for split_file in required_splits:
        if not (splits_dir / split_file).exists():
            print(f"❌ Required split file not found: {split_file}")
            return False
    
    print("✅ All requirements met")
    return True

def setup_experts():
    """Setup experts with fallback options"""
    print("\n🎯 Expert Setup Options:")
    print("1. Train RIDE experts from scratch (recommended, ~6-8 hours)")
    print("2. Use randomly initialized RIDE experts (fast, ~10 minutes)")
    print("3. Try pre-trained models (may fail due to download issues)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return False
    
    if choice == "1":
        print("\n🔄 Training RIDE experts from scratch...")
        return run_command([sys.executable, "-m", "src.train.train_expert"], 
                          "Training RIDE Experts from Scratch")
    
    elif choice == "2":
        print("\n🔄 Setting up randomly initialized RIDE experts...")
        return run_command([sys.executable, "-m", "src.utils.manual_pretrained_setup"], 
                          "Setting up Random RIDE Experts")
    
    elif choice == "3":
        print("\n🔄 Trying pre-trained models...")
        success = run_command([sys.executable, "-m", "src.train.train_expert", "--use-pretrained"], 
                             "Setting up Pre-trained RIDE Experts", check=False)
        
        if not success:
            print("\n⚠️ Pre-trained setup failed. Falling back to random initialization...")
            return run_command([sys.executable, "-m", "src.utils.manual_pretrained_setup"], 
                              "Fallback: Random RIDE Experts")
        return success
    
    else:
        print("Invalid choice. Defaulting to training from scratch...")
        return run_command([sys.executable, "-m", "src.train.train_expert"], 
                          "Training RIDE Experts from Scratch")

def main():
    """Main function"""
    print("🚀 RIDE-based AR-GSE Quick Start")
    print("=" * 60)
    print("This script helps you get started with the RIDE-based AR-GSE pipeline")
    print("with automatic fallback options if pre-trained models fail.")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please prepare data splits first.")
        return False
    
    # Setup experts
    if not setup_experts():
        print("\n❌ Expert setup failed. Cannot continue.")
        return False
    
    # Continue with pipeline
    print("\n🎯 Continuing with AR-GSE pipeline...")
    
    # Step 2: Train gating network
    if not run_command([sys.executable, "-m", "src.train.train_gating_only", "--mode", "selective"],
                      "Training Selective Gating Network"):
        return False
    
    # Step 3: Plugin optimization
    if not run_command([sys.executable, "run_improved_eg_outer.py"],
                      "GSE Plugin Optimization"):
        return False
    
    # Step 4: Evaluation
    if not run_command([sys.executable, "-m", "src.train.eval_gse_plugin"],
                      "Final Evaluation"):
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
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! Check the results in the output directories.")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")
    sys.exit(0 if success else 1)
