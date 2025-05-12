#!/usr/bin/env python3
import os
# Enable deterministic cuBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, confusion_matrix
from tqdm import tqdm

from models.feature_extractor import FeatureExtractor
from models.transformer_encoder import TransformerClassifier
from models.pointwise_conv_classifier import PointwiseConv1DClassifier
from data.RAVDESS_dataset import RAVDESSDataset

import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# cuDNN
torch.backends.cudnn.benchmark = False  # don’t autotune kernels
torch.backends.cudnn.deterministic = True  # force deterministic kernels
# PyTorch 1.8+
torch.use_deterministic_algorithms(True)

# If you use DataLoader with multiple workers, also:
os.environ["PYTHONHASHSEED"] = str(SEED)

def collate_fn(batch, max_seq_len, downsample_factor):
    waveforms, emos, gens, aids = zip(*batch)
    max_audio = max_seq_len * downsample_factor
    cropped = [(w[:max_audio] if w.shape[0] > max_audio else w) for w in waveforms]
    lengths = [w.shape[0] for w in cropped]
    max_len = max(lengths)
    B = len(cropped)
    padded = torch.zeros(B, max_len)
    for i, w in enumerate(cropped):
        padded[i, : w.shape[0]] = w
    emos = torch.tensor(emos, dtype=torch.long)
    gens = torch.tensor(gens, dtype=torch.long)
    aids = torch.tensor(aids, dtype=torch.long)
    return padded, emos, gens, aids

def compute_metrics(y_true, y_pred):
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    recall = recall_score(y_true, y_pred, average='macro')
    return acc, recall

def main(args):
    # checkpoint_dir is the timestamped logs folder
    out_dir = Path(args.checkpoint_dir)

    # load config from used_config.yaml inside that folder
    config_path = out_dir / "used_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'])

    # prepare log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"evaluation_{timestamp}.txt"

    # load speaker stats if needed
    speaker_norm = cfg['training'].get('speaker_wise_normalization', False)
    if speaker_norm:
        stats_file = Path(cfg['data']['root']) / f"speaker_feature_stats_{cfg['model']['type']}.json"
        with open(stats_file) as f:
            raw_stats = json.load(f)
        spk_means = {int(a): torch.tensor(v['mean'], device=device) for a, v in raw_stats.items()}
        spk_stds  = {int(a): torch.tensor(v['std'],  device=device) for a, v in raw_stats.items()}

    # dataset & loader
    test_ds = RAVDESSDataset(
        root=cfg['data']['root'],
        split="test",
        sample_rate=cfg.get('sample_rate', 16000)
    )
    down_f = 320
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, cfg['training']['max_seq_len'], down_f)
    )

    # load feature extractor
    fe = FeatureExtractor(
        model_type=cfg['model']['type'],
        pretrained_model_name_or_path=cfg['model']['pretrained'],
        trainable=False,
        use_weighted_sum=cfg['model'].get('use_weighted_sum', False)
    ).to(device)
    fe.eval()

    # load classifier
    clf_type = cfg['model'].get('classifier', 'transformer').lower()
    if clf_type == 'transformer':
        tconf = cfg['transformer']
        clf = TransformerClassifier(
            feature_dim=fe.model.config.hidden_size,
            input_dim=tconf['input_dim'],
            num_layers=tconf['num_layers'],
            nhead=tconf['nhead'],
            dim_feedforward=tconf['dim_feedforward'],
            num_emotions=cfg['training'].get('num_emotions', 7),
            num_genders=cfg['training'].get('num_genders', 2),
            dropout=tconf['dropout'],
            pool=tconf['pool'],
            speaker_wise_normalization=cfg['model'].get('speaker_wise_normalization', False)
        )
    else:
        conv_h = cfg['conv1d']['hidden_dim']
        conv_d = cfg['conv1d']['dropout']
        clf = PointwiseConv1DClassifier(
            input_dim=fe.model.config.hidden_size,
            hidden_dim=conv_h,
            num_emotions=cfg['training']['num_emotions'],
            dropout=conv_d,
            speaker_wise_normalization=cfg['model'].get('speaker_wise_normalization', False)
        )
    clf = clf.to(device)

    # Count learnable parameters and approximate model size
    fe_params = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    clf_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    total_params = fe_params + clf_params
    fe_size_mb = fe_params * 4 / (1024 ** 2)
    clf_size_mb = clf_params * 4 / (1024 ** 2)
    total_size_mb = total_params * 4 / (1024 ** 2)
    param_info = (
        f"Parameters - Fe: {fe_params:,}, Clf: {clf_params:,}, Total: {total_params:,}\n"
        f"Model Size   - Fe: {fe_size_mb:.2f} MB, Clf: {clf_size_mb:.2f} MB, Total: {total_size_mb:.2f} MB\n"
    )
    print(param_info)

    # Load the best saved models (feature extractor and classifier)
    fe_ckpt  = out_dir / "best_fe.pt"
    clf_ckpt = out_dir / "best_clf.pt"
    fe.load_state_dict(torch.load(fe_ckpt, map_location=device))
    clf.load_state_dict(torch.load(clf_ckpt, map_location=device))

    # evaluate
    all_true, all_pred = [], []
    all_genders = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="EVALUATE")
        for wave, emos, gens, aids in pbar:
            wave, emos = wave.to(device), emos.to(device)
            out = fe(wave)
            feats = out if isinstance(out, torch.Tensor) else out[-1]

            if clf_type == 'transformer':
                emo_logits, _ = clf(feats, lambda_grl=0.0)
            else:
                emo_logits = clf(feats)

            preds = emo_logits.argmax(dim=1)
            all_true.extend(emos.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_genders.extend(gens.cpu().tolist())

    # Compute per-gender metrics
    male_true = [t for t, g in zip(all_true, all_genders) if g == 0]
    male_pred = [p for p, g in zip(all_pred, all_genders) if g == 0]
    female_true = [t for t, g in zip(all_true, all_genders) if g == 1]
    female_pred = [p for p, g in zip(all_pred, all_genders) if g == 1]

    male_acc, male_recall = compute_metrics(male_true, male_pred)
    female_acc, female_recall = compute_metrics(female_true, female_pred)
    male_cm   = confusion_matrix(male_true, male_pred)
    female_cm = confusion_matrix(female_true, female_pred)

    acc, recall = compute_metrics(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)

    result = param_info + (
        f"Test Accuracy: {acc:.4f}\n"
        f"Test UAR    : {recall:.4f}\n"
        f"Confusion Matrix:\n{cm}"
    )
    # Print gender-specific results and confusion matrices
    result += (
        f"\n\nMale  - Accuracy: {male_acc:.4f}, UAR: {male_recall:.4f}\n"
        f"Male Confusion Matrix:\n{male_cm}\n\n"
        f"Female- Accuracy: {female_acc:.4f}, UAR: {female_recall:.4f}\n"
        f"Female Confusion Matrix:\n{female_cm}"
    )
    print(result)
    with open(log_path, "w") as f:
        f.write(result + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate FairSpeech model on RAVDESS test set"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to log/<timestamp> folder with used_config.yaml, best_fe.pt & best_clf.pt"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--num_workers", type=int, default=4
    )
    args = parser.parse_args()
    main(args)