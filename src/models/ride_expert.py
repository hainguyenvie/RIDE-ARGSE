"""
RIDE-based Expert Model for AR-GSE Pipeline
Adapted from RIDE paper: "Long-tailed Recognition by Routing Diverse Distribution-aware Experts"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, List


class NormedLinear(nn.Module):
    """Normalized Linear layer from RIDE"""
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):
    """Lambda layer to wrap functions as modules (for option A shortcut)"""
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Basic ResNet Block for CIFAR"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR ResNet paper uses option A
                # Use LambdaLayer instead of raw lambda for proper serialization
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def _weights_init(m):
    """Initialize weights"""
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class RIDEExpert(nn.Module):
    """
    RIDE-based Expert Model for CIFAR-100
    
    This model implements the RIDE architecture with:
    - Shared early layers (conv1, bn1, layer1)
    - Expert-specific later layers (layer2s, layer3s) 
    - Individual expert heads (linears)
    - Diversity-aware training support
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        num_experts: int = 3,
        reduce_dimension: bool = True,
        use_norm: bool = True,
        dropout_rate: float = 0.0,
        layer2_output_dim: Optional[int] = None,
        layer3_output_dim: Optional[int] = None,
        s: float = 30.0,
        init_weights: bool = True,
        **kwargs
    ):
        super(RIDEExpert, self).__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.reduce_dimension = reduce_dimension
        self.use_norm = use_norm
        self.s = s if use_norm else 1.0
        
        # Shared early layers
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)  # 5 blocks for ResNet32
        
        # Expert-specific layers
        self.in_planes = self.next_in_planes
        
        # Determine layer dimensions
        if layer2_output_dim is None:
            layer2_output_dim = 24 if reduce_dimension else 32
        if layer3_output_dim is None:
            layer3_output_dim = 48 if reduce_dimension else 64
            
        self.layer2_output_dim = layer2_output_dim
        self.layer3_output_dim = layer3_output_dim
        
        # Create expert-specific layers
        self.layer2s = nn.ModuleList([
            self._make_layer(BasicBlock, layer2_output_dim, 5, stride=2) 
            for _ in range(num_experts)
        ])
        
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([
            self._make_layer(BasicBlock, layer3_output_dim, 5, stride=2)
            for _ in range(num_experts)
        ])
        
        # Expert-specific classifiers
        if use_norm:
            self.linears = nn.ModuleList([
                NormedLinear(layer3_output_dim, num_classes) 
                for _ in range(num_experts)
            ])
        else:
            self.linears = nn.ModuleList([
                nn.Linear(layer3_output_dim, num_classes) 
                for _ in range(num_experts)
            ])
        
        # Dropout
        self.use_dropout = dropout_rate > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize weights
        if init_weights:
            self.apply(_weights_init)
        
        # Storage for features and logits (for RIDE loss)
        self.feat = []
        self.expert_logits = []

    def _make_layer(self, block, planes, num_blocks, stride):
        """Make a layer with specified blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        self.next_in_planes = self.in_planes
        
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)

    def _separate_part(self, x, expert_idx):
        """Process input through expert-specific layers"""
        out = x
        out = self.layer2s[expert_idx](out)
        out = self.layer3s[expert_idx](out)
        
        # Global Average Pooling
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        
        # Store feature for potential use
        self.feat.append(out)
        
        # Dropout
        if self.use_dropout:
            out = self.dropout(out)
        
        # Classification
        out = self.linears[expert_idx](out)
        out = out * self.s
        
        return out

    def forward(self, x, return_features=False):
        """
        Forward pass through RIDE expert model
        
        Args:
            x: Input tensor [B, 3, 32, 32]
            return_features: Whether to return features and individual logits
            
        Returns:
            If return_features=False: ensemble logits [B, num_classes]
            If return_features=True: dict with 'output', 'logits', 'feat'
        """
        # Shared early layers
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        # Expert-specific processing
        expert_outs = []
        self.feat = []
        self.expert_logits = []
        
        for expert_idx in range(self.num_experts):
            expert_out = self._separate_part(out, expert_idx)
            expert_outs.append(expert_out)
            self.expert_logits.append(expert_out)
        
        # Ensemble average
        final_out = torch.stack(expert_outs, dim=1).mean(dim=1)
        
        if return_features:
            return {
                'output': final_out,
                'logits': torch.stack(expert_outs, dim=1),  # [B, num_experts, num_classes]
                'feat': torch.stack(self.feat, dim=1) if self.feat else None  # [B, num_experts, feat_dim]
            }
        else:
            return final_out

    def get_calibrated_logits(self, x, temperature=1.0):
        """Get calibrated logits for evaluation"""
        logits = self.forward(x, return_features=False)
        return logits / temperature

    def summary(self):
        """Print model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"RIDEExpert Model Summary:")
        print(f"  • Architecture: ResNet32-like with {self.num_experts} experts")
        print(f"  • Classes: {self.num_classes}")
        print(f"  • Experts: {self.num_experts}")
        print(f"  • Layer2 dim: {self.layer2_output_dim}")
        print(f"  • Layer3 dim: {self.layer3_output_dim}")
        print(f"  • Use norm: {self.use_norm}")
        print(f"  • Dropout: {self.dropout.p if self.use_dropout else 0.0}")
        print(f"  • Total params: {total_params:,}")
        print(f"  • Trainable params: {trainable_params:,}")


class RIDELoss(nn.Module):
    """
    RIDE Loss combining individual expert losses with diversity regularization
    
    Adapted from RIDE paper for individual expert training in AR-GSE pipeline
    """
    
    def __init__(
        self,
        base_loss_factor: float = 1.0,
        diversity_factor: float = -0.2,
        diversity_temperature: float = 1.0,
        reweight: bool = True,
        reweight_epoch: int = 160,
        class_counts: Optional[List[int]] = None,
        **kwargs
    ):
        super(RIDELoss, self).__init__()
        
        self.base_loss_factor = base_loss_factor
        self.diversity_factor = diversity_factor
        self.diversity_temperature = diversity_temperature
        self.reweight = reweight
        self.reweight_epoch = reweight_epoch
        self.current_epoch = 0
        
        # Base loss (Cross Entropy)
        self.base_loss = nn.CrossEntropyLoss()
        
        # Class reweighting (optional)
        if class_counts is not None and reweight:
            effective_num = 1.0 - torch.pow(0.9999, torch.tensor(class_counts, dtype=torch.float32))
            weights = (1.0 - 0.9999) / effective_num
            per_cls_weights = weights / weights.sum() * len(weights)
            self.register_buffer('per_cls_weights', per_cls_weights)  # Use register_buffer for device management
        else:
            self.per_cls_weights = None

    def to(self, device):
        """Move module to device"""
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        return self
    
    def _hook_before_epoch(self, epoch):
        """Update epoch for reweighting schedule"""
        self.current_epoch = epoch

    def forward(self, model_output, target):
        """
        Compute RIDE loss
        
        Args:
            model_output: Output from RIDEExpert.forward(return_features=True)
            target: Ground truth labels [B]
            
        Returns:
            Total loss combining individual expert losses and diversity regularization
        """
        if isinstance(model_output, dict):
            ensemble_logits = model_output['output']  # [B, C]
            expert_logits = model_output['logits']    # [B, E, C]
        else:
            # Fallback for simple tensor output
            return self.base_loss(model_output, target)
        
        total_loss = 0.0
        num_experts = expert_logits.size(1)
        
        # Individual expert losses
        for expert_idx in range(num_experts):
            expert_output = expert_logits[:, expert_idx, :]  # [B, C]
            
            # Base loss for this expert
            if self.per_cls_weights is not None and self.current_epoch >= self.reweight_epoch:
                # Use class reweighting
                loss = F.cross_entropy(expert_output, target, weight=self.per_cls_weights)
            else:
                loss = self.base_loss(expert_output, target)
            
            total_loss += self.base_loss_factor * loss
            
            # Diversity regularization
            if self.diversity_factor != 0:
                # KL divergence between expert and ensemble
                expert_dist = F.log_softmax(expert_output / self.diversity_temperature, dim=1)
                ensemble_dist = F.softmax(ensemble_logits / self.diversity_temperature, dim=1)
                
                diversity_loss = F.kl_div(expert_dist, ensemble_dist, reduction='batchmean')
                total_loss += self.diversity_factor * diversity_loss
        
        return total_loss


# Alias for backward compatibility
Expert = RIDEExpert
