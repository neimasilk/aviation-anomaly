"""
Focal Loss for Class Imbalance

Reference: Lin et al. (2017) - Focal Loss for Dense Object Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples (usually minority class).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        gamma: float = 2.0,
        class_weights: List[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if class_weights is None:
            self.focal_weights = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        else:
            self.focal_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class indices
        """
        # Get weights for each target
        weights = self.focal_weights.to(inputs.device)
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        weights_t = weights.gather(0, targets)
        
        # Focal loss formula
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = weights_t * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
