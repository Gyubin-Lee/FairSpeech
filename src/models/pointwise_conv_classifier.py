import torch
import torch.nn as nn

class PointwiseConv1DClassifier(nn.Module):
    """
    A simple downstream emotion classifier.
    Internally uses two pointwise (kernel_size=1) Conv1D layers
    with ReLU + dropout, global average pooling, then a final
    linear layer to num_emotions.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_emotions: int = 7,
        dropout: float = 0.2
    ):
        super().__init__()
        # two pointwise conv layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.act1  = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.act2  = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # final classifier
        self.fc = nn.Linear(hidden_dim, num_emotions)

    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, seq_len, input_dim)
            src_key_padding_mask: ignored
            lambda_grl: ignored
        Returns:
            emo_logits: Tensor of shape (batch_size, num_emotions)
        """
        # rearrange to (batch, channels, seq_len)
        x = features.permute(0, 2, 1)  

        # first pointwise conv
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # second pointwise conv
        x = self.conv2(x)
        x = self.act2(x)
        x = self.drop2(x)

        # global average pooling over time dimension
        x = x.mean(dim=2)  # shape: (batch_size, hidden_dim)

        # final linear projection
        emo_logits = self.fc(x)  # shape: (batch_size, num_emotions)
        return emo_logits


# Example quick test
if __name__ == "__main__":
    batch, seq_len, dim = 4, 250, 768
    feats = torch.randn(batch, seq_len, dim)
    cls = PointwiseConv1DClassifier(input_dim=dim, hidden_dim=128, num_emotions=7)
    out = cls(feats)
    print("Output shape:", out.shape)  # should be (4, 7)