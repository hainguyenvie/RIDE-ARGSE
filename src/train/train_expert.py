import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import collections

# Import our custom modules
from src.models.ride_expert import RIDEExpert, RIDELoss
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
from src.metrics.calibration import TemperatureScaler
from src.data.dataloader_utils import get_expert_training_dataloaders

# --- EXPERT CONFIGURATIONS ---
# HYBRID APPROACH: 3 RIDE models with different loss functions
# Each model has 3 diverse experts (RIDE diversity) + different loss strategies (group specialization)
EXPERT_CONFIGS = {
    'ride_ce': {
        'name': 'ride_ce',
        'loss_type': 'ce',  # Standard CE (head-biased)
        'num_experts': 3,
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.0,
        'milestones': [160, 180],
        'gamma': 0.01,
        'warmup_epochs': 5,
        'diversity_factor': -0.2,  # Moderate diversity for CE
        'diversity_temperature': 1.0,
        'reweight': False,  # No reweight = head bias
        'reweight_epoch': 160
    },
    'ride_logitadjust': {
        'name': 'ride_logitadjust',
        'loss_type': 'logitadjust',  # LogitAdjust (balanced)
        'num_experts': 3,
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.0,
        'milestones': [160, 180],
        'gamma': 0.01,
        'warmup_epochs': 5,
        'diversity_factor': -0.45,  # Aggressive diversity for balance
        'diversity_temperature': 1.0,
        'reweight': True,  # With reweight
        'reweight_epoch': 160
    },
    'ride_balsoftmax': {
        'name': 'ride_balsoftmax',
        'loss_type': 'balsoftmax',  # BalancedSoftmax (tail-focused)
        'num_experts': 3,
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.0,
        'milestones': [160, 180],
        'gamma': 0.01,
        'warmup_epochs': 5,
        'diversity_factor': -0.35,  # Moderate diversity for tail
        'diversity_temperature': 1.0,
        'reweight': True,  # With reweight
        'reweight_epoch': 160
    }
}

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
        'num_groups': 2,
    },
    'train_params': {
        'batch_size': 128,
        'momentum': 0.9,
        'warmup_steps': 10,
    },
    'output': {
        'checkpoints_dir': './checkpoints/experts',
        'logits_dir': './outputs/logits',
    },
    'export': {
        'individual_experts': True,  # Export individual RIDE experts (9 total) instead of ensemble (3 total)
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def get_dataloaders():
    """Get train and validation dataloaders."""
    print("Loading CIFAR-100-LT datasets...")
    
    train_loader, val_loader = get_expert_training_dataloaders(
        batch_size=CONFIG['train_params']['batch_size'],
        num_workers=4
    )
    
    print(f"  Train loader: {len(train_loader)} batches ({len(train_loader.dataset):,} samples)")
    print(f"  Val loader: {len(val_loader)} batches ({len(val_loader.dataset):,} samples)")
    
    return train_loader, val_loader


def get_loss_function(loss_type, train_loader, expert_config):
    """Create RIDE loss function with diversity training and optional loss-specific strategies."""
    print(f"Creating {loss_type} loss with RIDE diversity...")
    
    # Get class counts from dataset
    if hasattr(train_loader.dataset, 'cifar_dataset'):
        train_targets = np.array(train_loader.dataset.cifar_dataset.targets)[train_loader.dataset.indices]
    elif hasattr(train_loader.dataset, 'dataset'):
        train_targets = np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices]
    else:
        train_targets = np.array(train_loader.dataset.targets)
    
    class_counts = [count for _, count in sorted(collections.Counter(train_targets).items())]
    
    # Create RIDELoss with appropriate base loss strategy
    if loss_type == 'ce':
        # Standard CE (no reweight) - head-biased
        return RIDELoss(
            diversity_factor=expert_config['diversity_factor'],
            diversity_temperature=expert_config['diversity_temperature'],
            reweight=False,  # No reweight for head bias
            reweight_epoch=expert_config.get('reweight_epoch', 160),
            class_counts=None  # No class counts needed for pure CE
        )
    elif loss_type in ['logitadjust', 'balsoftmax']:
        # LogitAdjust or BalancedSoftmax with reweighting
        return RIDELoss(
            diversity_factor=expert_config['diversity_factor'],
            diversity_temperature=expert_config['diversity_temperature'],
            reweight=expert_config.get('reweight', True),
            reweight_epoch=expert_config.get('reweight_epoch', 160),
            class_counts=class_counts  # Provide counts for reweighting
        )
    else:
        # Fallback
        return RIDELoss(
            diversity_factor=expert_config['diversity_factor'],
            diversity_temperature=expert_config['diversity_temperature'],
            reweight=expert_config.get('reweight', False),
            reweight_epoch=expert_config.get('reweight_epoch', 160),
            class_counts=class_counts if expert_config.get('reweight', False) else None
        )


def validate_model(model, val_loader, device):
    """Validate model with group-wise metrics."""
    model.eval()
    correct = 0
    total = 0
    
    group_correct = {'head': 0, 'tail': 0}
    group_total = {'head': 0, 'tail': 0}
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Group-wise accuracy (Head: 0-49, Tail: 50-99)
            for i, target in enumerate(targets):
                pred = predicted[i]
                if target < 50:  # Head classes
                    group_total['head'] += 1
                    if pred == target:
                        group_correct['head'] += 1
                else:  # Tail classes
                    group_total['tail'] += 1
                    if pred == target:
                        group_correct['tail'] += 1
    
    overall_acc = 100 * correct / total
    
    group_accs = {}
    for group in ['head', 'tail']:
        if group_total[group] > 0:
            group_accs[group] = 100 * group_correct[group] / group_total[group]
        else:
            group_accs[group] = 0.0
    
    return overall_acc, group_accs

def export_logits_for_all_splits(model, expert_name, export_individual_experts=True):
    """
    Export logits for all dataset splits.
    
    Args:
        model: RIDE expert model
        expert_name: Base name (e.g., 'ride_ce_expert')
        export_individual_experts: If True, export 3 individual experts from RIDE model
    """
    print(f"Exporting logits for expert '{expert_name}'...")
    if export_individual_experts:
        print(f"  🔬 Mode: Individual RIDE experts (will create {model.num_experts} variants)")
    else:
        print(f"  📊 Mode: Ensemble average")
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])

    # Define splits to export
    splits_info = [
        # From training set
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'tuneV', 'dataset_type': 'train', 'file': 'tuneV_indices.json'},
        {'name': 'val_small', 'dataset_type': 'train', 'file': 'val_small_indices.json'},
        # From test set  
        {'name': 'val_lt', 'dataset_type': 'test', 'file': 'val_lt_indices.json'},
        {'name': 'test_lt', 'dataset_type': 'test', 'file': 'test_lt_indices.json'},
        # Additional: calib split for CRC (if exists)
        {'name': 'calib', 'dataset_type': 'train', 'file': 'calib_indices.json'},
    ]
    
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
        if export_individual_experts:
            # Export individual expert logits (3 experts from this RIDE model)
            all_expert_logits = [[] for _ in range(model.num_experts)]
            
            with torch.no_grad():
                for inputs, _ in tqdm(loader, desc=f"Exporting {split_name} (individual)", leave=False):
                    model_output = model(inputs.to(DEVICE), return_features=True)
                    expert_logits = model_output['logits']  # [B, num_experts, C]
                    
                    for expert_idx in range(model.num_experts):
                        all_expert_logits[expert_idx].append(expert_logits[:, expert_idx, :].cpu())
            
            # Save each individual expert
            for expert_idx in range(model.num_experts):
                expert_logits_tensor = torch.cat(all_expert_logits[expert_idx])
                
                # Create directory for this specific expert
                individual_expert_name = f"{expert_name}_expert_{expert_idx}"
                individual_output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / individual_expert_name
                individual_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = individual_output_dir / f"{split_name}_logits.pt"
                torch.save(expert_logits_tensor.to(torch.float16), output_file)
            
            print(f"  ✅ {split_name}: {len(indices):,} samples → {model.num_experts} individual experts")
        else:
            # Export ensemble average (old behavior)
            output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            all_logits = []
            with torch.no_grad():
                for inputs, _ in tqdm(loader, desc=f"Exporting {split_name}"):
                    logits = model.get_calibrated_logits(inputs.to(DEVICE))
                    all_logits.append(logits.cpu())
            
            all_logits = torch.cat(all_logits)
            torch.save(all_logits.to(torch.float16), output_dir / f"{split_name}_logits.pt")
            print(f"  ✅ {split_name}: {len(indices):,} samples → ensemble average")
    
    if export_individual_experts:
        print(f"✅ Exported {model.num_experts} individual experts from {expert_name}")
    else:
        print(f"✅ Exported ensemble average for {expert_name}")

# --- CORE TRAINING FUNCTIONS ---

def train_single_expert(expert_key):
    """Train a single expert based on its configuration."""
    if expert_key not in EXPERT_CONFIGS:
        raise ValueError(f"Expert '{expert_key}' not found in EXPERT_CONFIGS")
    
    expert_config = EXPERT_CONFIGS[expert_key]
    expert_name = expert_config['name']
    loss_type = expert_config['loss_type']
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING EXPERT: {expert_name.upper()}")
    print(f"🎯 Loss Type: {loss_type.upper()}")
    print(f"{'='*60}")
    
    # Setup
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    train_loader, val_loader = get_dataloaders()
    
    # Model - TRUE RIDE with num_experts from config
    num_experts = expert_config.get('num_experts', 9)  # Default 9 for max diversity
    print(f"🔬 Creating RIDE model with {num_experts} diverse experts...")
    
    model = RIDEExpert(
        num_classes=CONFIG['dataset']['num_classes'],
        num_experts=num_experts,  # From config (9 for true RIDE)
        reduce_dimension=True,  # RIDE standard for CIFAR
        use_norm=True,  # RIDE uses normalized linear
        dropout_rate=expert_config['dropout_rate'],
        init_weights=True
    ).to(DEVICE)
    
    criterion = get_loss_function(loss_type, train_loader, expert_config)
    criterion = criterion.to(DEVICE)  # Move criterion to device
    print(f"✅ Loss Function: {type(criterion).__name__}")
    
    # Print model summary
    print("📊 Model Architecture:")
    model.summary()
    
    # Optimizer and scheduler - RIDE style
    optimizer = optim.SGD(
        model.parameters(), 
        lr=expert_config['lr'],
        momentum=CONFIG['train_params']['momentum'],
        weight_decay=expert_config['weight_decay'],
        nesterov=True  # RIDE uses Nesterov
    )
    
    # RIDE-style scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = expert_config.get('warmup_epochs', 5)
        milestones = expert_config['milestones']
        gamma = expert_config['gamma']
        
        # Determine base LR multiplier
        lr_mult = 1.0
        for milestone in milestones:
            if epoch >= milestone:
                lr_mult *= gamma
        
        # Apply warmup
        if epoch < warmup_epochs:
            lr_mult = lr_mult * float(1 + epoch) / warmup_epochs
            
        return lr_mult
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training setup
    best_acc = 0.0
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"best_{expert_name}.pth"
    
    # Training loop
    for epoch in range(expert_config['epochs']):
        # Train
        model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{expert_config['epochs']}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # RIDE forward pass with features
            model_output = model(inputs, return_features=True)
            
            # Update loss function epoch for reweighting
            if hasattr(criterion, '_hook_before_epoch'):
                criterion._hook_before_epoch(epoch)
            
            loss = criterion(model_output, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        val_acc, group_accs = validate_model(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1:3d}: Loss={running_loss/len(train_loader):.4f}, "
            f"Val Acc={val_acc:.2f}%, Head={group_accs['head']:.1f}%, "
            f"Tail={group_accs['tail']:.1f}%, LR={scheduler.get_last_lr()[0]:.5f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"💾 New best! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
    
    # Post-training: Calibration
    print(f"\n--- 🔧 POST-PROCESSING: {expert_name} ---")
    model.load_state_dict(torch.load(best_model_path))
    
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(model, val_loader, DEVICE)
    model.set_temperature(optimal_temp)
    print(f"✅ Temperature calibration: T = {optimal_temp:.3f}")
    
    final_model_path = checkpoint_dir / f"final_calibrated_{expert_name}.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Final validation
    final_acc, final_group_accs = validate_model(model, val_loader, DEVICE)
    print(f"📊 Final Results - Overall: {final_acc:.2f}%, "
        f"Head: {final_group_accs['head']:.1f}%, "
        f"Tail: {final_group_accs['tail']:.1f}%")
    
    # Export logits
    export_logits_for_all_splits(model, expert_name, CONFIG['export']['individual_experts'])
    
    print(f"✅ COMPLETED: {expert_name}")
    return final_model_path


def main():
    """Main training script - trains TRUE RIDE model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TRUE RIDE Expert Training')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pre-trained RIDE checkpoint instead of training from scratch')
    parser.add_argument('--pretrained-path', type=str, default=None,
                       help='Path to pre-trained RIDE checkpoint file (.pth). '
                            'For CIFAR-100-LT, use the 3-expert model from MODEL_ZOO')
    parser.add_argument('--num-experts', type=int, default=None,
                       help='Override number of experts to export (default: use config value)')
    parser.add_argument('--pretrained-model', 
                       choices=['ride_standard', 'ride_distill', 'ride_distill_4experts'],
                       default='ride_standard',
                       help='Which pre-trained model to download (if --pretrained-path not provided)')
    
    args = parser.parse_args()
    
    if args.use_pretrained:
        print("="*60)
        print("🔄 TRUE RIDE: Using Pre-trained Checkpoint")
        print("="*60)
        
        # Determine number of experts to export
        num_experts_to_export = args.num_experts if args.num_experts else EXPERT_CONFIGS['ride_ensemble']['num_experts']
        print(f"Will export {num_experts_to_export} experts from checkpoint")
        
        try:
            # Use local checkpoint path if provided, otherwise try download
            if args.pretrained_path:
                from src.utils.load_pretrained import setup_from_checkpoint_path
                print(f"📁 Loading from local path: {args.pretrained_path}")
                print(f"📊 This should be a RIDE checkpoint with 3+ experts")
                print(f"   Download from: https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view")
                print()
                
                # The checkpoint likely has 3 experts, we'll export what we need
                expert_names = setup_from_checkpoint_path(
                    args.pretrained_path, 
                    DEVICE,
                    num_experts=3,  # CIFAR-100-LT RIDE model has 3 experts
                    export_individual=True,
                    target_num_experts=num_experts_to_export
                )
            else:
                from src.utils.download_ride_pretrained import setup_pretrained_experts
                print(f"📥 Downloading pre-trained model: {args.pretrained_model}")
                expert_names = setup_pretrained_experts(args.pretrained_model, DEVICE)
            
            if expert_names:
                print("\n" + "="*60)
                print("✅ Pre-trained TRUE RIDE experts setup completed!")
                print("="*60)
                print(f"Created {len(expert_names)} experts from ONE pre-trained model:")
                for name in expert_names:
                    print(f"  • {name}")
                print()
                print("Next step: Train gating network")
                print("python -m src.train.train_gating_only --mode selective")
                print("="*60)
                return  # Exit successfully
            else:
                print("❌ Failed to setup pre-trained experts")
                raise Exception("Setup returned empty expert list")
                
        except Exception as e:
            print(f"❌ Error setting up pre-trained experts: {e}")
            print("\n" + "="*60)
            print("🔧 ALTERNATIVES:")
            print("="*60)
            print("1. Use local checkpoint file (RECOMMENDED):")
            print("   Download RIDE checkpoint manually, then:")
            print("   python -m src.train.train_expert --use-pretrained --pretrained-path /path/to/checkpoint.pth")
            print("   Download from: https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view")
            print()
            print("2. Train RIDE experts from scratch:")
            print("   python -m src.train.train_expert")
            print()
            print("3. Use randomly initialized RIDE experts (fast fallback):")
            print("   python -m src.utils.manual_pretrained_setup")
            print()
            print("4. Install gdown for automatic downloads:")
            print("   pip install gdown")
            print("   Then retry: python -m src.train.train_expert --use-pretrained")
            print("="*60)
            
            # Ask user what to do
            print("\nWhat would you like to do?")
            print("1. Train from scratch (recommended)")
            print("2. Use random initialization (faster)")
            print("3. Exit and try manual download")
            
            try:
                choice = input("Enter choice (1-3): ").strip()
                if choice == "2":
                    print("\n🔄 Setting up randomly initialized experts...")
                    from src.utils.manual_pretrained_setup import create_random_initialized_experts
                    expert_names = create_random_initialized_experts(DEVICE)
                    if expert_names:
                        print("✅ Random experts setup completed!")
                        return
                elif choice == "3":
                    print("Exiting. Please download manually and retry.")
                    return
                else:
                    print("Falling back to training from scratch...")
                    args.use_pretrained = False
            except KeyboardInterrupt:
                print("\nExiting...")
                return
    
    if not args.use_pretrained:
        print("🚀 HYBRID RIDE Expert Training Pipeline")
        print(f"Device: {DEVICE}")
        print(f"Dataset: {CONFIG['dataset']['name']}")
        print(f"Strategy: 3 RIDE models × 3 experts = 9 total experts")
        print(f"  - RIDE-CE (head-biased, no reweight)")
        print(f"  - RIDE-LogitAdjust (balanced, with reweight)")
        print(f"  - RIDE-BalancedSoftmax (tail-focused, with reweight)")
        print(f"Key: Diversity within + Specialization across models!")
        
        results = {}
        
        for expert_key in EXPERT_CONFIGS.keys():
            try:
                model_path = train_single_expert(expert_key)
                results[expert_key] = {'status': 'success', 'path': model_path}
            except Exception as e:
                print(f"❌ Failed to train {expert_key}: {e}")
                results[expert_key] = {'status': 'failed', 'error': str(e)}
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print("🏁 TRAINING SUMMARY")
        print(f"{'='*60}")
        
        for expert_key, result in results.items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {expert_key}: {result['status']}")
            if result['status'] == 'failed':
                print(f"    Error: {result['error']}")
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\nSuccessfully trained {successful}/{len(EXPERT_CONFIGS)} models")
        print(f"Total experts: {successful * 3}")
        
        if successful > 0:
            print(f"\nNext step: Train gating network")
            print(f"python -m src.train.train_gating_only --mode selective")


if __name__ == '__main__':
    main()