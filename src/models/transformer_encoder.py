# src/models/transformer_encoder.py

import torch
import torch.nn as nn
from .grl import GradientReversal
from .heads import EmotionHead, GenderHead

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for Speech Emotion Recognition with
    adversarial gender debiasing, optionally using a learnable [CLS] token.

    Args:
        feature_dim (int): output hidden_size of the feature extractor
        input_dim (int): desired d_model for Transformer (e.g. 768)
        num_layers (int): number of TransformerEncoderLayers
        nhead (int): number of attention heads
        dim_feedforward (int): feed-forward network dimension
        num_emotions (int): number of emotion classes
        num_genders (int): number of gender classes (usually 2)
        dropout (float): dropout rate in Transformer layers
        pool (str): pooling strategy: 'mean', 'max', or 'cls'
    """
    def __init__(
        self,
        feature_dim: int,
        input_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        num_emotions: int = 7,
        num_genders: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",
        speaker_wise_normalization: bool = False,
        predict_gender: bool = False,
    ):
        super().__init__()

        # input projection if needed
        if feature_dim != input_dim:
            self.input_proj = nn.Linear(feature_dim, input_dim)
        else:
            self.input_proj = nn.Identity()

        # learnable positional embeddings
        self.max_len = 250

        # learnable [CLS] token for 'cls' pooling
        self.pool = pool
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        else:
            self.cls_token = None

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification heads
        self.emotion_head = EmotionHead(input_dim, num_emotions)
        self.predict_gender = predict_gender
        if self.predict_gender:
            self.grl = GradientReversal()
            self.gender_head = GenderHead(input_dim, num_genders)

        # optional channel-wise LayerNorm over the hidden dimension
        self.speaker_wise_normalization = speaker_wise_normalization
        if speaker_wise_normalization:
            self.channel_norm = nn.LayerNorm(feature_dim)
        else:
            self.channel_norm = nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        lambda_grl: float = 1.0
    ):
        """
        Args:
            features: (batch_size, seq_len, feature_dim)
            src_key_padding_mask: (batch_size, seq_len), True for masked
            lambda_grl: float, strength of gradient reversal
        Returns:
            emo_logits: (batch_size, num_emotions)
            gender_logits: (batch_size, num_genders)
        """

        # apply channel-wise LayerNorm if enabled
        features = self.channel_norm(features)

        # project features to transformer dimension
        x = self.input_proj(features)  # (batch, seq_len, input_dim)

        if self.pool == "cls":
            # prepend [CLS] token
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch,1,dim)
            x = torch.cat([cls_tokens,x], dim=1)                   # (batch, seq_len+1, dim)
            self.max_len += 1  # adjust max_len for [CLS] token

        # reshape for transformer: (seq_len(+1), batch, dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.permute(1, 0, 2)  # back to (batch, seq_len(+1), dim)

        # pooling
        if self.pool == "mean":
            pooled = x.mean(dim=1)
        elif self.pool == "max":
            pooled = x.max(dim=1)[0]
        elif self.pool == "cls":
            pooled = x[:, 0]  # first token is [CLS]
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")

        # emotion prediction
        emo_logits = self.emotion_head(pooled)
        gender_logits = None
        # adversarial gender prediction
        if self.predict_gender:
            grl_feat = self.grl(pooled, lambda_grl)
            gender_logits = self.gender_head(grl_feat)

        return emo_logits, gender_logits