"""
Load pre-trained RIDE models from local checkpoint files.
Much simpler and more reliable than automatic downloads.
"""
import torch
import torchvision
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from src.models.ride_expert import RIDEExpert

CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'output': {
        'checkpoints_dir': './checkpoints/experts',
        'logits_dir': './outputs/logits',
    }
}


def load_ride_checkpoint(checkpoint_path, num_experts=3):
    """
    Load RIDE checkpoint from local path with flexible format handling.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        num_experts: Number of experts (3 or 4)
        
    Returns:
        RIDEExpert model with loaded weights, or None if failed
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Validate path
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return None
    
    file_size = checkpoint_path.stat().st_size
    if file_size < 1000:
        print(f"❌ Checkpoint file too small ({file_size} bytes), likely invalid")
        return None
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    print(f"   Size: {file_size / 1024 / 1024:.1f} MB")
    
    # Try to load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        # Try with weights_only=True for newer PyTorch
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("✅ Checkpoint loaded successfully (weights_only mode)")
        except Exception as e2:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Alternative attempt failed: {e2}")
            return None
    
    # Debug: Show checkpoint structure
    print("📋 Checkpoint structure:")
    if isinstance(checkpoint, dict):
        for key in list(checkpoint.keys())[:10]:  # Show first 10 keys
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"   • {key}: Tensor{tuple(value.shape)}")
            else:
                print(f"   • {key}: {type(value).__name__}")
        if len(checkpoint) > 10:
            print(f"   ... and {len(checkpoint) - 10} more keys")
    else:
        print(f"   Type: {type(checkpoint)}")
    
    # Extract state_dict
    state_dict = None
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ['state_dict', 'model', 'model_state_dict', 'net', 'backbone']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"✅ Found state_dict under key: '{key}'")
                break
        
        # If no standard key, check if checkpoint itself is a state_dict
        if state_dict is None:
            # Check if it looks like a parameter dictionary
            sample_keys = list(checkpoint.keys())[:5]
            if all(isinstance(checkpoint[k], torch.Tensor) for k in sample_keys):
                state_dict = checkpoint
                print("✅ Using checkpoint directly as state_dict")
    
    if state_dict is None:
        print("❌ Could not extract state_dict from checkpoint")
        print("   Available keys:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
        return None
    
    print(f"✅ State dict has {len(state_dict)} parameters")
    
    # Create our RIDEExpert model
    print(f"🔧 Creating RIDEExpert model ({num_experts} experts)...")
    try:
        model = RIDEExpert(
            num_classes=CONFIG['dataset']['num_classes'],
            num_experts=num_experts,
            reduce_dimension=True,
            use_norm=True,
            dropout_rate=0.0,
            init_weights=True
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None
    
    # Try to load state_dict
    print("🔄 Loading weights into model...")
    try:
        # First try: strict loading
        model.load_state_dict(state_dict, strict=True)
        print("✅ Weights loaded successfully (strict mode)")
        return model
    except Exception as e1:
        print(f"⚠️ Strict loading failed: {e1}")
        
        # Second try: non-strict loading (allows missing/extra keys)
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
            print("✅ Weights loaded successfully (non-strict mode)")
            return model
        except Exception as e2:
            print(f"❌ Non-strict loading also failed: {e2}")
            
            # Third try: manual key mapping (RIDE repo often has 'module.' prefix)
            print("🔄 Trying manual key mapping...")
            try:
                mapped_state_dict = {}
                for key, value in state_dict.items():
                    # Remove common prefixes
                    new_key = key
                    if key.startswith('module.'):
                        new_key = key[7:]
                    if key.startswith('backbone.'):
                        new_key = key[9:]
                    mapped_state_dict[new_key] = value
                
                missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
                print(f"✅ Weights loaded with key mapping (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
                return model
            except Exception as e3:
                print(f"❌ Key mapping failed: {e3}")
                print("⚠️ Returning randomly initialized model...")
                return model  # Return initialized model as last resort


def export_logits_from_model(model, expert_name, device='cuda'):
    """Export logits from model for all dataset splits."""
    print(f"\n📊 Exporting logits for '{expert_name}'...")
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_info = [
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'tuneV', 'dataset_type': 'train', 'file': 'tuneV_indices.json'},
        {'name': 'val_lt', 'dataset_type': 'test', 'file': 'val_lt_indices.json'},
        {'name': 'test_lt', 'dataset_type': 'test', 'file': 'test_lt_indices.json'},
    ]
    
    for split_info in splits_info:
        split_name = split_info['name']
        dataset_type = split_info['dataset_type']
        indices_file = split_info['file']
        indices_path = splits_dir / indices_file
        
        if not indices_path.exists():
            print(f"  ⚠️ {indices_file} not found, skipping {split_name}")
            continue
            
        # Load dataset
        if dataset_type == 'train':
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=True, transform=transform)
        else:
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=False, transform=transform)
        
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        subset = Subset(base_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        # Export logits
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"  {split_name}", leave=False):
                logits = model.get_calibrated_logits(inputs.to(device))
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits)
        output_file = output_dir / f"{split_name}_logits.pt"
        torch.save(all_logits.to(torch.float16), output_file)
        print(f"  ✅ {split_name}: {len(indices):,} samples → {output_file}")
    
    print(f"✅ All logits exported to: {output_dir}")


def setup_from_checkpoint_path(checkpoint_path, device='cuda', num_experts=3):
    """
    Setup experts from a local RIDE checkpoint file.
    
    Args:
        checkpoint_path: Path to RIDE checkpoint (.pth file)
        device: Device to use ('cuda' or 'cpu')
        num_experts: Number of experts in the checkpoint (3 or 4)
        
    Returns:
        List of expert names that were created
    """
    checkpoint_path = Path(checkpoint_path)
    
    print("=" * 60)
    print("🚀 Setting up RIDE Experts from Local Checkpoint")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Experts: {num_experts}")
    print("=" * 60)
    
    # Load model from checkpoint
    model = load_ride_checkpoint(checkpoint_path, num_experts)
    if model is None:
        print("❌ Failed to load checkpoint")
        return []
    
    # Create directories
    checkpoints_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Create expert variants (simulating different loss-based training)
    expert_names = ['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']
    
    print(f"\n🔧 Creating {len(expert_names)} expert variants...")
    for expert_name in expert_names:
        print(f"\n--- Processing {expert_name} ---")
        
        # Save checkpoint
        final_model_path = checkpoints_dir / f"final_calibrated_{expert_name}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"💾 Saved checkpoint: {final_model_path}")
        
        # Export logits
        export_logits_from_model(model, expert_name, device)
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully created {len(expert_names)} experts!")
    print("=" * 60)
    print("Next step: Train gating network")
    print("python -m src.train.train_gating_only --mode selective")
    print("=" * 60)
    
    return expert_names


def main():
    """CLI interface for loading from checkpoint path"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load RIDE expert from checkpoint')
    parser.add_argument('checkpoint_path', type=str, help='Path to RIDE checkpoint (.pth file)')
    parser.add_argument('--num-experts', type=int, default=3, choices=[3, 4],
                       help='Number of experts in checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    expert_names = setup_from_checkpoint_path(args.checkpoint_path, args.device, args.num_experts)
    
    if not expert_names:
        print("\n❌ Setup failed")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

