import json
from pathlib import Path
from collections import defaultdict

import torch
import librosa
from tqdm import tqdm

from src.models.feature_extractor import FeatureExtractor

def compute_speaker_feature_stats(
    data_root: str,
    model_type: str = "wav2vec2",
    pretrained_model: str = "facebook/wav2vec2-base-960h",
    device: str = "cuda",
    sr: int = 16000
):
    """
    Computes per-speaker mean and std over the final-layer features
    extracted by the FeatureExtractor on all WAV files under data_root.
    Saves a JSON file mapping actor_id -> {"mean": [...], "std": [...]}.
    The filename includes the model_type for clarity.
    """
    root = Path(data_root)
    device = torch.device(device)

    # Initialize feature extractor
    fe = FeatureExtractor(
        model_type=model_type,
        pretrained_model_name_or_path=pretrained_model,
        trainable=False,
        use_weighted_sum=False
    ).to(device)
    fe.eval()
    feature_dim = fe.model.config.hidden_size

    # Prepare accumulators
    sums   = defaultdict(lambda: torch.zeros(feature_dim, device=device))
    sumsqs = defaultdict(lambda: torch.zeros(feature_dim, device=device))
    counts = defaultdict(int)

    print("Gathering feature sums per speaker...")
    for gender_dir in root.iterdir():
        if not gender_dir.is_dir():
            continue
        for actor_dir in sorted(gender_dir.iterdir()):
            if not actor_dir.is_dir():
                continue
            actor_id = int(actor_dir.name.split("_")[-1])
            for emo_dir in actor_dir.iterdir():
                if not emo_dir.is_dir():
                    continue
                for wav_path in emo_dir.glob("*.wav"):
                    wav_np, _ = librosa.load(str(wav_path), sr=sr, mono=True)
                    waveform = torch.from_numpy(wav_np).unsqueeze(0).to(device)
                    with torch.no_grad():
                        hidden_states = fe(waveform)
                    feats = hidden_states[-1].squeeze(0)
                    sums[actor_id]   += feats.sum(dim=0)
                    sumsqs[actor_id] += (feats ** 2).sum(dim=0)
                    counts[actor_id] += feats.shape[0]

    # Compute mean and std per speaker
    stats = {}
    print("Computing mean/std and writing JSON...")
    for aid, sum_vec in sums.items():
        n = counts[aid]
        mean_vec = (sum_vec / n).cpu().tolist()
        var_vec  = (sumsqs[aid] / n - (sum_vec / n) ** 2).cpu().tolist()
        std_vec  = [max(v, 1e-6) ** 0.5 for v in var_vec]
        stats[aid] = {"mean": mean_vec, "std": std_vec}

    # Save to JSON with model_type in filename
    filename = f"speaker_feature_stats_{model_type}.json"
    out_path = root / filename
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved speaker stats to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute per-speaker feature normalization stats"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to the processed RAVDESS root directory"
    )
    parser.add_argument(
        "--model_type", type=str, default="wav2vec2",
        choices=["wav2vec2", "hubert"]
    )
    parser.add_argument(
        "--pretrained_model", type=str,
        default="facebook/wav2vec2-base-960h"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Audio sampling rate"
    )
    args = parser.parse_args()
    compute_speaker_feature_stats(
        data_root=args.data_root,
        model_type=args.model_type,
        pretrained_model=args.pretrained_model,
        device=args.device,
        sr=args.sample_rate
    )