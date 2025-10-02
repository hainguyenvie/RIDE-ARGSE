#!/usr/bin/env python3
"""
Analyze individual expert quality to understand why α is stuck at 1.0
"""
import torch
import torchvision
import numpy as np
import json
from pathlib import Path

def analyze_expert_logits():
    """Analyze quality and diversity of individual RIDE experts"""
    
    print("="*60)
    print("Analyzing RIDE Expert Quality")
    print("="*60)
    
    # Load test data
    splits_dir = Path('data/cifar100_lt_if100_splits')
    with open(splits_dir / 'test_lt_indices.json') as f:
        test_indices = json.load(f)
    
    cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    test_labels = torch.tensor(np.array(cifar_test.targets)[test_indices])
    
    # Define groups (head: classes 0-49, tail: classes 50-99)
    is_head = (test_labels < 50)
    is_tail = (test_labels >= 50)
    
    print(f"\nTest set: {len(test_labels)} samples")
    print(f"  Head: {is_head.sum().item()} samples")
    print(f"  Tail: {is_tail.sum().item()} samples")
    
    # Load expert logits
    logits_root = Path('outputs/logits/cifar100_lt_if100')
    expert_names = ['ride_ensemble_expert_0', 'ride_ensemble_expert_1', 'ride_ensemble_expert_2']
    
    print(f"\n{'='*60}")
    print("Individual Expert Performance")
    print(f"{'='*60}")
    
    expert_logits_list = []
    for expert_name in expert_names:
        logit_path = logits_root / expert_name / 'test_lt_logits.pt'
        if not logit_path.exists():
            print(f"❌ Missing: {logit_path}")
            return
        
        logits = torch.load(logit_path, map_location='cpu', weights_only=False).float()
        expert_logits_list.append(logits)
        
        # Compute accuracy
        preds = logits.argmax(dim=1)
        overall_acc = (preds == test_labels).float().mean().item()
        head_acc = (preds[is_head] == test_labels[is_head]).float().mean().item()
        tail_acc = (preds[is_tail] == test_labels[is_tail]).float().mean().item()
        
        # Compute confidence
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1)[0]
        avg_conf = max_probs.mean().item()
        head_conf = max_probs[is_head].mean().item()
        tail_conf = max_probs[is_tail].mean().item()
        
        print(f"\n{expert_name}:")
        print(f"  Accuracy:   Overall={overall_acc:.3f}, Head={head_acc:.3f}, Tail={tail_acc:.3f}")
        print(f"  Confidence: Overall={avg_conf:.3f}, Head={head_conf:.3f}, Tail={tail_conf:.3f}")
    
    # Analyze diversity
    print(f"\n{'='*60}")
    print("Expert Diversity Analysis")
    print(f"{'='*60}")
    
    all_logits = torch.stack(expert_logits_list, dim=1)  # [B, 3, C]
    all_probs = torch.softmax(all_logits, dim=-1)
    all_preds = all_logits.argmax(dim=-1)  # [B, 3]
    
    # Prediction diversity
    for i in range(len(expert_names)):
        for j in range(i+1, len(expert_names)):
            disagreement = (all_preds[:, i] != all_preds[:, j]).float().mean().item()
            print(f"  Expert {i} vs Expert {j}: {disagreement:.1%} disagreement")
    
    # Average pairwise KL divergence
    print(f"\nPairwise KL Divergence (measures RIDE diversity):")
    for i in range(len(expert_names)):
        for j in range(i+1, len(expert_names)):
            kl = torch.sum(all_probs[:, i, :] * (torch.log(all_probs[:, i, :] + 1e-8) - torch.log(all_probs[:, j, :] + 1e-8)), dim=-1)
            avg_kl = kl.mean().item()
            print(f"  Expert {i} vs Expert {j}: KL={avg_kl:.4f}")
    
    # Ensemble performance
    print(f"\n{'='*60}")
    print("Ensemble Performance")
    print(f"{'='*60}")
    
    ensemble_logits = all_logits.mean(dim=1)  # [B, C]
    ensemble_preds = ensemble_logits.argmax(dim=1)
    ensemble_acc = (ensemble_preds == test_labels).float().mean().item()
    ensemble_head_acc = (ensemble_preds[is_head] == test_labels[is_head]).float().mean().item()
    ensemble_tail_acc = (ensemble_preds[is_tail] == test_labels[is_tail]).float().mean().item()
    
    print(f"\nSimple Average Ensemble:")
    print(f"  Overall: {ensemble_acc:.3f}")
    print(f"  Head: {ensemble_head_acc:.3f}")
    print(f"  Tail: {ensemble_tail_acc:.3f}")
    
    # Compare to individual experts
    best_individual_acc = max([
        (all_preds[:, i] == test_labels).float().mean().item() 
        for i in range(len(expert_names))
    ])
    print(f"\nBest individual expert: {best_individual_acc:.3f}")
    print(f"Ensemble improvement: {ensemble_acc - best_individual_acc:+.3f}")
    
    # Group-specific analysis
    print(f"\n{'='*60}")
    print("Group-Specific Expert Performance")
    print(f"{'='*60}")
    
    for expert_idx, expert_name in enumerate(expert_names):
        preds = all_preds[:, expert_idx]
        head_acc = (preds[is_head] == test_labels[is_head]).float().mean().item()
        tail_acc = (preds[is_tail] == test_labels[is_tail]).float().mean().item()
        ratio = head_acc / (tail_acc + 1e-6)
        print(f"\n{expert_name}:")
        print(f"  Head acc: {head_acc:.3f}")
        print(f"  Tail acc: {tail_acc:.3f}")
        print(f"  Head/Tail ratio: {ratio:.2f}x")
    
    # Check if there's any specialization
    head_accs = [(all_preds[:, i][is_head] == test_labels[is_head]).float().mean().item() for i in range(3)]
    tail_accs = [(all_preds[:, i][is_tail] == test_labels[is_tail]).float().mean().item() for i in range(3)]
    
    print(f"\n{'='*60}")
    print("Specialization Check")
    print(f"{'='*60}")
    print(f"Head accuracies: {[f'{x:.3f}' for x in head_accs]}")
    print(f"Tail accuracies: {[f'{x:.3f}' for x in tail_accs]}")
    print(f"Head variance: {np.var(head_accs):.6f}")
    print(f"Tail variance: {np.var(tail_accs):.6f}")
    
    if np.var(head_accs) < 0.0001 and np.var(tail_accs) < 0.0001:
        print("\nWARNING: Experts are nearly identical!")
        print("   This means RIDE's diversity training didn't work or")
        print("   the checkpoint doesn't have diverse experts.")
        print("\nRecommendation: Train from scratch with TRUE RIDE")
        print("   python -m src.train.train_expert")
    else:
        print("\nExperts show some diversity")
        print("   RIDE diversity is working!")

if __name__ == '__main__':
    analyze_expert_logits()

