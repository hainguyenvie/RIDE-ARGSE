"""
Manual setup for pre-trained RIDE experts.
This provides an alternative when automatic download/conversion fails.
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

def create_random_initialized_experts(device='cuda'):
    """
    Create randomly initialized RIDE experts.
    This is a fallback when pre-trained weights can't be loaded.
    """
    print("Creating randomly initialized RIDE experts...")
    
    # Create directories
    checkpoints_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Create expert names
    expert_names = ['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']
    
    # Create and save models
    for expert_name in expert_names:
        print(f"Creating {expert_name}...")
        
        model = RIDEExpert(
            num_classes=CONFIG['dataset']['num_classes'],
            num_experts=3,
            reduce_dimension=True,
            use_norm=True,
            dropout_rate=0.0,
            init_weights=True
        ).to(device)
        
        # Save checkpoint
        final_model_path = checkpoints_dir / f"final_calibrated_{expert_name}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Saved: {final_model_path}")
        
        # Export logits
        export_logits_from_model(model, expert_name)
    
    print(f"✅ Created {len(expert_names)} randomly initialized experts")
    return expert_names

def export_logits_from_model(model, expert_name):
    """Export logits from model for all dataset splits."""
    print(f"Exporting logits for '{expert_name}'...")
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define splits to export
    splits_info = [
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'tuneV', 'dataset_type': 'train', 'file': 'tuneV_indices.json'},
        {'name': 'val_lt', 'dataset_type': 'test', 'file': 'val_lt_indices.json'},
        {'name': 'test_lt', 'dataset_type': 'test', 'file': 'test_lt_indices.json'},
    ]
    
    device = next(model.parameters()).device
    
    for split_info in splits_info:
        split_name = split_info['name']
        dataset_type = split_info['dataset_type']
        indices_file = split_info['file']
        indices_path = splits_dir / indices_file
        
        if not indices_path.exists():
            print(f"  Warning: {indices_file} not found, skipping {split_name}")
            continue
            
        # Load appropriate base dataset
        if dataset_type == 'train':
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=True, transform=transform)
        else:
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=False, transform=transform)
        
        # Load indices and create subset
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        subset = Subset(base_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        # Export logits
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {split_name}"):
                logits = model.get_calibrated_logits(inputs.to(device))
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), output_dir / f"{split_name}_logits.pt")
        print(f"  Exported {split_name}: {len(indices):,} samples")
    
    print(f"✅ Logits exported to: {output_dir}")

def main():
    """Main function for manual setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual RIDE expert setup')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("🔧 Manual RIDE Expert Setup")
    print("=" * 60)
    print("This creates randomly initialized RIDE experts as a fallback")
    print("when pre-trained weights cannot be downloaded/loaded.")
    print("=" * 60)
    
    expert_names = create_random_initialized_experts(args.device)
    
    if expert_names:
        print("\n✅ Manual setup completed!")
        print("Note: These are randomly initialized (not pre-trained) experts.")
        print("They will still use the RIDE architecture but without pre-trained weights.")
        print("\nYou can now proceed to:")
        print("python -m src.train.train_gating_only --mode selective")
    else:
        print("\n❌ Manual setup failed.")

if __name__ == '__main__':
    main()
