#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel

class FeatureExtractor(nn.Module):
    """
    Wrapper for HuggingFace Wav2Vec2 or HuBERT models that extracts
    hidden representations (all layers) from raw audio.
    """
    def __init__(
        self,
        model_type: str = "wav2vec2",
        pretrained_model_name_or_path: str = "facebook/wav2vec2-base-960h",
        trainable: bool = False
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

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_values: (batch, seq_len) raw waveform, float32 in [-1,1]
            attention_mask: (batch, seq_len) optional
        Returns:
            Tuple of hidden_states: (embeddings, layer1, ..., layerN)
        """
        outputs = self.model(input_values, attention_mask=attention_mask)
        return outputs.hidden_states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test FeatureExtractor with a single audio file"
    )
    parser.add_argument(
        "--model_type",
        choices=["wav2vec2", "hubert"],
        default="wav2vec2",
        help="Which pretrained model to use"
    )
    parser.add_argument(
        "--pretrained_model",
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace model identifier or path"
    )
    parser.add_argument(
        "--audio_path",
        required=True,
        help="Path to a .wav audio file"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (cpu or cuda)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    fe = FeatureExtractor(
        model_type=args.model_type,
        pretrained_model_name_or_path=args.pretrained_model,
        trainable=False
    ).to(device)
    fe.eval()

    # Load audio file
    import librosa
    waveform_np, sample_rate = librosa.load(args.audio_path, sr=None)

    # Convert to torch.Tensor with shape (1, seq_len)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    # Normalize to float32 [-1,1]
    if waveform.dtype != torch.float32:
        waveform = waveform.to(torch.float32)
        max_val = waveform.abs().max().item()
        if max_val > 0:
            waveform /= max_val

    input_values = waveform.to(device)

    with torch.no_grad():
        hidden_states = fe(input_values)

    print(f"Number of layers (including embeddings): {len(hidden_states)}")
    for i, layer in enumerate(hidden_states):
        print(f" Layer {i:2d}: shape {tuple(layer.shape)}") # (batch, seq_len, hidden_size)