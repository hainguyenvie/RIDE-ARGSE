# src/models/gating.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingFeatureBuilder:
    """
    Builds scalable, class-count-independent features from expert posteriors/logits.
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @torch.no_grad()
    def __call__(self, expert_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_logits: Tensor of shape [B, E, C] (Batch, Experts, Classes)
        
        Returns:
            A feature tensor of shape [B, D] where D is the feature dimension.
        """
        # Ensure input is float32 for stable calculations
        expert_logits = expert_logits.float()
        B, E, C = expert_logits.shape
        
        # Use posteriors for probability-based features
        expert_posteriors = torch.softmax(expert_logits, dim=-1)

        # Feature 1: Entropy of each expert's prediction  [B, E]
        entropy = -torch.sum(expert_posteriors * torch.log(expert_posteriors + 1e-8), dim=-1)

        # Feature 2: Top-k probability mass and residual mass per expert
        topk_vals, _ = torch.topk(expert_posteriors, k=min(self.top_k, expert_posteriors.size(-1)), dim=-1)
        topk_mass = torch.sum(topk_vals, dim=-1)            # [B, E]
        residual_mass = 1.0 - topk_mass                    # [B, E]

        # Feature 3: Expert confidence (max prob) and top1-top2 gap
        max_probs, _ = expert_posteriors.max(dim=-1)        # [B, E]
        if topk_vals.size(-1) >= 2:
            top1 = topk_vals[..., 0]
            top2 = topk_vals[..., 1]
            top_gap = top1 - top2                           # [B, E]
        else:
            top_gap = torch.zeros_like(max_probs)

        # Feature 4: Cosine similarity to ensemble mean posterior (agreement proxy)
        mean_posterior = torch.mean(expert_posteriors, dim=1)        # [B, C]
        cosine_sim = F.cosine_similarity(expert_posteriors, mean_posterior.unsqueeze(1), dim=-1)  # [B, E]

        # Feature 5: KL divergence of each expert to mean posterior (disagreement)
        # KL(p_e || mean_p) = Σ p_e (log p_e - log mean_p)
        kl_to_mean = torch.sum(expert_posteriors * (torch.log(expert_posteriors + 1e-8) - torch.log(mean_posterior.unsqueeze(1) + 1e-8)), dim=-1)  # [B, E]

        # Global (per-sample) features capturing ensemble uncertainty / disagreement
        # Mixture / mean posterior entropy
        mean_entropy = -torch.sum(mean_posterior * torch.log(mean_posterior + 1e-8), dim=-1)  # [B]
        # Mean variance across classes (how much experts disagree on class probs)
        class_var = expert_posteriors.var(dim=1)                     # [B, C]
        mean_class_var = class_var.mean(dim=-1)                      # [B]
        # Std of expert max probabilities (confidence dispersion)
        std_max_conf = max_probs.std(dim=-1)                         # [B]

        # ========== NEW RIDE-SPECIFIC FEATURES ==========
        # These leverage RIDE's diversity training to improve routing
        
        # Feature 6: Expert disagreement on top prediction (crucial for RIDE!)
        top_classes = expert_posteriors.argmax(dim=-1)  # [B, E] - each expert's prediction
        mode_class = torch.mode(top_classes, dim=1)[0]  # [B] - most common prediction
        disagree_with_mode = (top_classes != mode_class.unsqueeze(1)).float()  # [B, E]
        
        # Feature 7: Confidence on ensemble's prediction (alignment)
        ensemble_top_class = mean_posterior.argmax(dim=-1)  # [B]
        confidence_on_ensemble = torch.gather(
            expert_posteriors, 2, 
            ensemble_top_class.unsqueeze(1).unsqueeze(2).expand(B, E, 1)
        ).squeeze(-1)  # [B, E]
        
        # Feature 8: Pairwise diversity (RIDE's diversity loss effect)
        # Measures how much experts actually disagree
        pairwise_kl_list = []
        for i in range(E):
            for j in range(i+1, E):
                kl_ij = torch.sum(
                    expert_posteriors[:, i, :] * (
                        torch.log(expert_posteriors[:, i, :] + 1e-8) - 
                        torch.log(expert_posteriors[:, j, :] + 1e-8)
                    ), dim=-1
                )
                pairwise_kl_list.append(kl_ij)
        mean_pairwise_kl = torch.stack(pairwise_kl_list, dim=1).mean(dim=1) if pairwise_kl_list else torch.zeros(B, device=expert_logits.device)  # [B]
        
        # Feature 9: Specialization gap (is there a clear specialist?)
        max_conf_per_sample = max_probs.max(dim=-1)[0]  # [B]
        mean_conf_per_sample = max_probs.mean(dim=-1)  # [B]
        specialization_gap = max_conf_per_sample - mean_conf_per_sample  # [B]
        
        # Feature 10: Overall prediction uncertainty (for selective prediction)
        prediction_uncertainty = kl_to_mean.mean(dim=-1)  # [B]
        
        # Concatenate per-expert features (all [B,E])
        per_expert_feats = [
            entropy, topk_mass, residual_mass, max_probs, top_gap, 
            cosine_sim, kl_to_mean, disagree_with_mode, confidence_on_ensemble
        ]
        per_expert_concat = torch.cat(per_expert_feats, dim=1)       # [B, 9*E]

        # Concatenate global features -> shape [B, 9E + 6]
        global_feats = torch.stack([
            mean_entropy, mean_class_var, std_max_conf,
            mean_pairwise_kl, specialization_gap, prediction_uncertainty
        ], dim=1)  # [B, 6]
        
        features = torch.cat([per_expert_concat, global_feats], dim=1)
        
        return features

class GatingNet(nn.Module):
    """
    A simple MLP that takes gating features and outputs expert weights.
    """
    def __init__(self, in_dim: int, hidden_dims: list = [128, 64], num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Gating features of shape [B, D]
        
        Returns:
            Expert weights (before softmax) of shape [B, E]
        """
        return self.net(x)