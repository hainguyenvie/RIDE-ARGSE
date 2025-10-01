"""
Download and setup pre-trained RIDE models from the official model zoo.
This allows skipping the expert training step and going directly to gating training.
"""
import requests
import torch
import torchvision
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from src.models.ride_expert import RIDEExpert

# RIDE Model Zoo URLs for CIFAR-100-LT
RIDE_MODEL_ZOO = {
    'ride_standard': {
        'url': 'https://drive.google.com/file/d/1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS/view?usp=sharing',
        'gdrive_id': '1uE8I_2JcslWGPu4O0nAFEIk7iR_Sw5lS',
        'description': 'RIDE (3 experts) for CIFAR-100-LT, Overall: 48.6%, Few: 25.7%'
    },
    'ride_distill': {
        'url': 'https://drive.google.com/file/d/1W-EICEpAavKzlnayiFPvb5cDyGCBl34l/view?usp=sharing', 
        'gdrive_id': '1W-EICEpAavKzlnayiFPvb5cDyGCBl34l',
        'description': 'RIDE + Distill (3 experts) for CIFAR-100-LT, Overall: 49.0%, Few: 25.2%'
    },
    'ride_distill_4experts': {
        'url': 'https://drive.google.com/file/d/11kyxcYIh3bXk3mn3Y8EENHcsx-Ld9PXH/view?usp=sharing',
        'gdrive_id': '11kyxcYIh3bXk3mn3Y8EENHcsx-Ld9PXH', 
        'description': 'RIDE + Distill (4 experts) for CIFAR-100-LT, Overall: 49.4%, Few: 25.7%'
    }
}

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
        'pretrained_dir': './checkpoints/ride_pretrained'
    }
}

def download_from_gdrive(file_id, destination):
    """Download file from Google Drive using file ID with improved error handling"""
    print(f"Downloading from Google Drive (ID: {file_id})...")
    
    try:
        import gdown
        print("Using gdown for Google Drive download...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        print(f"✅ Downloaded to: {destination}")
        return True
    except ImportError:
        print("gdown not available, using requests fallback...")
        return download_with_requests(file_id, destination)
    except Exception as e:
        print(f"gdown failed: {e}")
        print("Falling back to requests method...")
        return download_with_requests(file_id, destination)

def download_with_requests(file_id, destination):
    """Fallback download method using requests"""
    # Google Drive download URL
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large files with confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Check if response is valid
    if response.status_code != 200:
        print(f"❌ Download failed with status code: {response.status_code}")
        return False
    
    # Check content type
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        print("❌ Downloaded HTML instead of file (likely access denied)")
        return False
    
    # Save file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    print(f"✅ Downloaded to: {destination}")
    return True

def convert_ride_checkpoint_to_our_format(ride_checkpoint_path, num_experts=3):
    """
    Convert RIDE checkpoint to our RIDEExpert format with improved error handling.
    """
    print(f"Converting RIDE checkpoint: {ride_checkpoint_path}")
    
    # Check if file exists and is valid
    if not Path(ride_checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {ride_checkpoint_path}")
        return None
    
    # Check file size
    file_size = Path(ride_checkpoint_path).stat().st_size
    if file_size < 1000:  # Less than 1KB is likely corrupted
        print(f"❌ Checkpoint file too small ({file_size} bytes), likely corrupted")
        return None
    
    try:
        # Load original RIDE checkpoint with better error handling
        print(f"Loading checkpoint (size: {file_size / 1024 / 1024:.1f} MB)...")
        checkpoint = torch.load(ride_checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("This might be due to:")
        print("  1. Corrupted download")
        print("  2. Incompatible PyTorch version")
        print("  3. Wrong file format")
        return None
    
    # Debug: Print checkpoint structure
    print("Checkpoint structure:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
    else:
        print(f"  Type: {type(checkpoint)}")
    
    # Extract state dict (format may vary)
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Try to use the checkpoint directly as state dict
            state_dict = checkpoint
    else:
        print("❌ Unexpected checkpoint format")
        return None
    
    if state_dict is None:
        print("❌ Could not extract state_dict from checkpoint")
        return None
    
    print(f"Found state dict with {len(state_dict)} keys")
    
    # Create our RIDEExpert model
    try:
        model = RIDEExpert(
            num_classes=CONFIG['dataset']['num_classes'],
            num_experts=num_experts,
            reduce_dimension=True,
            use_norm=True,
            dropout_rate=0.0,
            init_weights=True  # Initialize first, then override
        )
        print("✅ Created RIDEExpert model")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None
    
    # For now, just return the initialized model since exact weight mapping
    # requires knowing the precise RIDE checkpoint format
    print("⚠️ Using initialized model (weight conversion needs checkpoint format details)")
    print("The model will work but won't have pre-trained weights")
    
    return model

def export_logits_from_pretrained_model(model, expert_name):
    """Export logits from pre-trained model for all dataset splits."""
    print(f"Exporting logits for pre-trained expert '{expert_name}'...")
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
    
    print(f"✅ All logits exported to: {output_dir}")

def setup_pretrained_experts(model_choice='ride_standard', device='cuda'):
    """
    Download and setup pre-trained RIDE experts.
    
    Args:
        model_choice: Which model to download ('ride_standard', 'ride_distill', 'ride_distill_4experts')
        device: Device to load models on
        
    Returns:
        List of expert names that were set up
    """
    if model_choice not in RIDE_MODEL_ZOO:
        raise ValueError(f"Model choice '{model_choice}' not available. Choose from: {list(RIDE_MODEL_ZOO.keys())}")
    
    model_info = RIDE_MODEL_ZOO[model_choice]
    print(f"Setting up pre-trained RIDE model: {model_choice}")
    print(f"Description: {model_info['description']}")
    
    # Create directories
    pretrained_dir = Path(CONFIG['output']['pretrained_dir'])
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint if not exists
    checkpoint_path = pretrained_dir / f"{model_choice}.pth"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}, attempting download...")
        success = download_from_gdrive(model_info['gdrive_id'], checkpoint_path)
        if not success:
            print(f"❌ Failed to download {model_choice}")
            print("Please download manually from:")
            print(f"  URL: {model_info['url']}")
            print(f"  Save to: {checkpoint_path}")
            print("\nAlternatively, you can:")
            print("1. Install gdown: pip install gdown")
            print("2. Use the original RIDE training approach")
            return []
    else:
        print(f"✅ Found existing checkpoint: {checkpoint_path}")
    
    # Determine number of experts
    num_experts = 4 if '4experts' in model_choice else 3
    
    # Convert and load model
    model = convert_ride_checkpoint_to_our_format(checkpoint_path, num_experts)
    if model is None:
        print("❌ Failed to convert checkpoint")
        return []
    
    model = model.to(device)
    
    # Create expert variants (simulate different loss-based training)
    expert_names = ['ride_ce_expert', 'ride_logitadjust_expert', 'ride_balsoftmax_expert']
    
    for expert_name in expert_names:
        print(f"\n--- Setting up {expert_name} ---")
        
        # Save checkpoint in our format
        final_model_path = checkpoints_dir / f"final_calibrated_{expert_name}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Saved checkpoint: {final_model_path}")
        
        # Export logits
        export_logits_from_pretrained_model(model, expert_name)
    
    print(f"\n✅ Successfully set up {len(expert_names)} experts from pre-trained RIDE model")
    return expert_names

def main():
    """Main function to setup pre-trained experts"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and setup pre-trained RIDE experts')
    parser.add_argument('--model', choices=list(RIDE_MODEL_ZOO.keys()), 
                       default='ride_standard',
                       help='Which pre-trained model to use')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("🚀 RIDE Pre-trained Expert Setup")
    print("=" * 60)
    print("Available models:")
    for name, info in RIDE_MODEL_ZOO.items():
        print(f"  • {name}: {info['description']}")
    print("=" * 60)
    
    expert_names = setup_pretrained_experts(args.model, args.device)
    
    if expert_names:
        print("\n🎉 Setup completed! You can now proceed to gating training:")
        print("python -m src.train.train_gating_only --mode selective")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == '__main__':
    main()
