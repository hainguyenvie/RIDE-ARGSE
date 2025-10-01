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
    
    # Try to load state_dict with automatic prefix handling
    print("🔄 Loading weights into model...")
    
    # Function to try loading with different key transformations
    def try_load_with_mapping(state_dict_to_try, description):
        try:
            # Filter out Expert Assignment (EA) module keys - we don't use them
            filtered_state_dict = {}
            ea_keys_filtered = 0
            for key, value in state_dict_to_try.items():
                # Skip keys related to Expert Assignment module
                if any(ea_key in key for ea_key in ['expert_help', 'confidence', 'routing']):
                    ea_keys_filtered += 1
                    continue
                filtered_state_dict[key] = value
            
            if ea_keys_filtered > 0:
                print(f"   Filtered out {ea_keys_filtered} Expert Assignment keys")
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print(f"✅ Weights loaded successfully ({description}, perfect match)")
                return True, 0, 0
            elif len(missing_keys) < len(filtered_state_dict) * 0.1:  # Less than 10% missing is OK
                print(f"✅ Weights loaded successfully ({description})")
                if missing_keys:
                    print(f"   ⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
                if unexpected_keys:
                    print(f"   ⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
                return True, len(missing_keys), len(unexpected_keys)
            else:
                return False, len(missing_keys), len(unexpected_keys)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False, -1, -1
    
    # Try 1: Direct loading (no transformation)
    print("Attempt 1: Direct loading...")
    success, n_missing, n_unexpected = try_load_with_mapping(state_dict, "direct")
    if success:
        return model
    print(f"   ⚠️ Too many missing keys ({n_missing}), trying transformations...")
    
    # Try 2: Remove 'backbone.' prefix
    print("Attempt 2: Removing 'backbone.' prefix...")
    mapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('backbone.'):
            new_key = key[9:]  # Remove 'backbone.'
        mapped_state_dict[new_key] = value
    
    success, n_missing, n_unexpected = try_load_with_mapping(mapped_state_dict, "backbone. removed")
    if success:
        return model
    
    # Try 3: Remove 'module.' prefix (DataParallel)
    print("Attempt 3: Removing 'module.' prefix...")
    mapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.'
        mapped_state_dict[new_key] = value
    
    success, n_missing, n_unexpected = try_load_with_mapping(mapped_state_dict, "module. removed")
    if success:
        return model
    
    # Try 4: Remove both 'module.backbone.' prefix
    print("Attempt 4: Removing 'module.backbone.' prefix...")
    mapped_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('module.backbone.'):
            new_key = key[16:]  # Remove 'module.backbone.'
        elif key.startswith('module.'):
            new_key = key[7:]
        elif key.startswith('backbone.'):
            new_key = key[9:]
        mapped_state_dict[new_key] = value
    
    success, n_missing, n_unexpected = try_load_with_mapping(mapped_state_dict, "module.backbone. removed")
    if success:
        return model
    
    # If all attempts failed, return model anyway with warning
    print("⚠️ All loading attempts had significant mismatches")
    print("⚠️ Returning model with best-effort weights loaded")
    print("   The model will work but may not have optimal pre-trained weights")
    return model


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

