#!/usr/bin/env python3
"""
Quick verification script to check if all files are in the right places.
Run this after each step to verify setup.
"""
from pathlib import Path
import json

def check_mark(condition):
    return "✅" if condition else "❌"

def verify_data_splits():
    """Verify data splits exist"""
    print("\n" + "="*60)
    print("📂 Checking Data Splits")
    print("="*60)
    
    splits_dir = Path("./data/cifar100_lt_if100_splits")
    required_files = ["train_indices.json", "tuneV_indices.json", "val_lt_indices.json", "test_lt_indices.json"]
    
    all_good = True
    for file in required_files:
        exists = (splits_dir / file).exists()
        print(f"{check_mark(exists)} {file}")
        if not exists:
            all_good = False
    
    return all_good

def verify_expert_logits():
    """Verify expert logits exist"""
    print("\n" + "="*60)
    print("📊 Checking Expert Logits")
    print("="*60)
    
    logits_dir = Path("./outputs/logits/cifar100_lt_if100")
    expert_names = ['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']
    required_splits = ['train_logits.pt', 'tuneV_logits.pt', 'val_lt_logits.pt', 'test_lt_logits.pt']
    
    all_good = True
    for expert_name in expert_names:
        expert_dir = logits_dir / expert_name
        expert_exists = expert_dir.exists()
        print(f"\n{expert_name}:")
        print(f"  {check_mark(expert_exists)} Directory exists")
        
        if expert_exists:
            for split in required_splits:
                split_exists = (expert_dir / split).exists()
                print(f"  {check_mark(split_exists)} {split}")
                if not split_exists:
                    all_good = False
        else:
            all_good = False
            print(f"  ⚠️ Run: python -m src.train.train_expert --use-pretrained --pretrained-path <path>")
    
    return all_good

def verify_gating_checkpoint():
    """Verify gating checkpoint exists"""
    print("\n" + "="*60)
    print("🧠 Checking Gating Checkpoint")
    print("="*60)
    
    gating_dir = Path("./checkpoints/gating_pretrained/cifar100_lt_if100")
    ckpt_file = gating_dir / "gating_selective.ckpt"
    logs_file = gating_dir / "selective_training_logs.json"
    
    ckpt_exists = ckpt_file.exists()
    logs_exist = logs_file.exists()
    
    print(f"{check_mark(ckpt_exists)} gating_selective.ckpt")
    print(f"{check_mark(logs_exist)} selective_training_logs.json")
    
    if not ckpt_exists:
        print("⚠️ Run: python -m src.train.train_gating_only --mode selective")
    
    return ckpt_exists

def verify_plugin_checkpoint():
    """Verify plugin checkpoint exists"""
    print("\n" + "="*60)
    print("🔌 Checking Plugin Checkpoint")
    print("="*60)
    
    plugin_file = Path("./checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt")
    exists = plugin_file.exists()
    
    print(f"{check_mark(exists)} gse_balanced_plugin.ckpt")
    
    if not exists:
        print("⚠️ Run: python run_improved_eg_outer.py")
    
    return exists

def verify_results():
    """Verify evaluation results exist"""
    print("\n" + "="*60)
    print("📈 Checking Evaluation Results")
    print("="*60)
    
    results_dir = Path("./results_worst_eg_improved/cifar100_lt_if100")
    required_files = ['metrics.json', 'rc_curve.csv', 'rc_curve_comparison.png']
    
    all_good = True
    for file in required_files:
        exists = (results_dir / file).exists()
        print(f"{check_mark(exists)} {file}")
        if not exists:
            all_good = False
    
    if not all_good:
        print("⚠️ Run: python -m src.train.eval_gse_plugin")
    
    return all_good

def main():
    """Run all verifications"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify AR-GSE pipeline setup')
    parser.add_argument('--step', type=int, choices=[0, 1, 2, 3, 4], default=0,
                       help='Verify specific step (0=all, 1=experts, 2=gating, 3=plugin, 4=results)')
    
    args = parser.parse_args()
    
    print("🔍 AR-GSE Pipeline Verification")
    print("="*60)
    
    results = {}
    
    # Always check data splits
    results['data_splits'] = verify_data_splits()
    
    if args.step == 0 or args.step == 1:
        results['expert_logits'] = verify_expert_logits()
    
    if args.step == 0 or args.step == 2:
        results['gating'] = verify_gating_checkpoint()
    
    if args.step == 0 or args.step == 3:
        results['plugin'] = verify_plugin_checkpoint()
    
    if args.step == 0 or args.step == 4:
        results['results'] = verify_results()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    for component, status in results.items():
        print(f"{check_mark(status)} {component.replace('_', ' ').title()}")
    
    all_good = all(results.values())
    
    if all_good:
        print("\n🎉 All checks passed!")
    else:
        print("\n⚠️ Some components missing. Follow the suggestions above.")
        print("\n📚 Full pipeline:")
        print("  1. python -m src.train.train_expert --use-pretrained --pretrained-path <path>")
        print("  2. python -m src.train.train_gating_only --mode selective")
        print("  3. python run_improved_eg_outer.py")
        print("  4. python -m src.train.eval_gse_plugin")
    
    return 0 if all_good else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
