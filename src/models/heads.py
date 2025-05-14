import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionHead(nn.Module):
    """
    Simple classification head for emotion prediction.

    Args:
        input_dim (int): dimensionality of the pooled features
        num_emotions (int): number of emotion classes
        dropout (float): dropout probability before the final linear layer
    """
    def __init__(self, input_dim: int, num_emotions: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_emotions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (batch_size, input_dim)
        Returns:
            logits (Tensor): shape (batch_size, num_emotions)
        """
        return self.classifier(x)

class GenderHead(nn.Module):
    """
    Simple classification head for gender prediction.

    Args:
        input_dim (int): dimensionality of the pooled features
        num_genders (int): number of gender classes (usually 2)
        dropout (float): dropout probability before the final linear layer
    """
    def __init__(self, input_dim: int, num_genders: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_genders)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (batch_size, input_dim)
        Returns:
            logits (Tensor): shape (batch_size, num_genders)
        """
        return self.classifier(x)