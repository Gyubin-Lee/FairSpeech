#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel

class FeatureExtractor(nn.Module):
    """
    Wrapper for HuggingFace Wav2Vec2 or HuBERT models that either:
      - returns all layer hidden_states (as a tuple), or
      - if use_weighted_sum=True, returns a single weighted average of them.

    Args:
        model_type (str): "wav2vec2" or "hubert"
        pretrained_model_name_or_path (str): HF model ID or local path
        trainable (bool): if False, freeze all backbone weights
        use_weighted_sum (bool): if True, learn layer weights α_i and return
                                 weighted sum:
                                     f = (Σ α_i f_i) / (Σ α_i)
    """
    def __init__(
        self,
        model_type: str = "wav2vec2",
        pretrained_model_name_or_path: str = "facebook/wav2vec2-base-960h",
        trainable: bool = False,
        use_weighted_sum: bool = False,
    ):
        super().__init__()
        model_type = model_type.lower()
        if model_type == "wav2vec2":
            self.model = Wav2Vec2Model.from_pretrained(
                pretrained_model_name_or_path,
                output_hidden_states=True,
                return_dict=True
            )
        elif model_type == "hubert":
            self.model = HubertModel.from_pretrained(
                pretrained_model_name_or_path,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

        self.use_weighted_sum = use_weighted_sum
        if self.use_weighted_sum:
            # Actually, HF hidden_states length = num_hidden_layers + 1
            num_layers = self.model.config.num_hidden_layers + 1
            # learnable α_i, initialize all to 1.0
            self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_values: (batch_size, seq_len) raw waveform
            attention_mask: (batch_size, seq_len) optional
        Returns:
            If use_weighted_sum:
                Tensor (batch_size, seq_len, hidden_size)
            else:
                Tuple of Tensors (num_layers, batch_size, seq_len, hidden_size)
        """
        outputs = self.model(input_values, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple length = num_hidden_layers+1

        if not self.use_weighted_sum:
            return hidden_states

        # stack to (num_layers, batch, seq_len, hidden)
        hs = torch.stack(hidden_states, dim=0)
        # weights α_i shaped (num_layers, 1, 1, 1)
        w = self.layer_weights.view(-1, 1, 1, 1)
        # weighted sum divided by sum α_i
        weighted = (w * hs).sum(dim=0) / self.layer_weights.sum()
        return weighted


if __name__ == "__main__":
    import argparse
    import librosa

    parser = argparse.ArgumentParser(description="Test weighted FeatureExtractor")
    parser.add_argument("--model_type", choices=["wav2vec2","hubert"], default="wav2vec2")
    parser.add_argument("--pretrained_model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--audio_path", required=True, help="Path to .wav file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_weighted_sum", action="store_true",
                        help="Use learnable weighted average of all layers")
    args = parser.parse_args()

    device = torch.device(args.device)
    fe = FeatureExtractor(
        model_type=args.model_type,
        pretrained_model_name_or_path=args.pretrained_model,
        trainable=False,
        use_weighted_sum=args.use_weighted_sum
    ).to(device)
    fe.eval()

    waveform_np, sr = librosa.load(args.audio_path, sr=None, mono=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = fe(waveform)
    if args.use_weighted_sum:
        print("Weighted features shape:", feats.shape)  # (batch, seq_len, hidden)
    else:
        print("Number of layers:", len(feats))
        for i, layer in enumerate(feats):
            print(f" Layer {i}: shape {tuple(layer.shape)}")