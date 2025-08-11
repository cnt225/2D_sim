import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Simple Mean Squared Error Loss for velocity prediction
    """
    def __init__(self, weight=1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 6] predicted twist vector
            target: [B, 6] target twist vector
        
        Returns:
            loss: scalar tensor
        """
        return self.weight * self.mse(pred, target)
