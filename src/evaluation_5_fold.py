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

import statistics

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

    # prepare log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"evaluation_{timestamp}.txt"
    results_txt = out_dir / "evaluation_results.txt"
    if results_txt.exists():
        results_txt.unlink()

    # Cross-fold evaluation
    fold_accs, fold_recs = [], []
    fold_male_accs, fold_male_recs = [], []
    fold_female_accs, fold_female_recs = [], []
    for fold_idx in range(1, 6):
        print(f"\n=== Evaluating fold {fold_idx} ===")
        fold_dir = out_dir / f"fold_{fold_idx}"

        # load config for this fold
        fold_cfg_path = fold_dir / "used_config.yaml"
        if not fold_cfg_path.exists():
            raise FileNotFoundError(f"Config not found at {fold_cfg_path}")
        with open(fold_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        num_folds = cfg['training'].get('num_folds', 5)
        device = torch.device(cfg['device'])

        clf_type = cfg['model'].get('classifier', 'transformer').lower()
        speaker_norm = cfg['model'].get('speaker_wise_normalization', False)

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
                speaker_wise_normalization=speaker_norm,
                predict_gender=tconf.get('predict_gender', False)
            )
        else:
            conv_h = cfg['conv1d']['hidden_dim']
            conv_d = cfg['conv1d']['dropout']
            clf = PointwiseConv1DClassifier(
                input_dim=fe.model.config.hidden_size,
                hidden_dim=conv_h,
                num_emotions=cfg['training']['num_emotions'],
                dropout=conv_d,
                speaker_wise_normalization=speaker_norm
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

        # load checkpoints for this fold
        fe.load_state_dict(torch.load(fold_dir/"best_fe.pt", map_location=device))
        clf.load_state_dict(torch.load(fold_dir/"best_clf.pt", map_location=device))

        all_true, all_pred, all_genders = [], [], []
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"EVALUATE Fold{fold_idx}")
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

        # overall metrics
        acc, rec = compute_metrics(all_true, all_pred)
        fold_accs.append(acc)
        fold_recs.append(rec)

        # per-gender metrics
        male_t = [t for t,g in zip(all_true, all_genders) if g==0]
        male_p = [p for p,g in zip(all_pred, all_genders) if g==0]
        female_t = [t for t,g in zip(all_true, all_genders) if g==1]
        female_p = [p for p,g in zip(all_pred, all_genders) if g==1]
        male_acc, male_rec = compute_metrics(male_t, male_p)
        female_acc, female_rec = compute_metrics(female_t, female_p)
        male_cm = confusion_matrix(male_t, male_p)
        female_cm = confusion_matrix(female_t, female_p)

        # accumulate per-fold gender metrics
        fold_male_accs.append(male_acc)
        fold_male_recs.append(male_rec)
        fold_female_accs.append(female_acc)
        fold_female_recs.append(female_rec)

        # print and log
        result = (
            f"Fold {fold_idx}:\n"
            f"  Overall Acc: {acc:.4f}, UAR: {rec:.4f}\n"
            f"  Male   Acc: {male_acc:.4f}, UAR: {male_rec:.4f}\n"
            f"  Male CM:\n{male_cm}\n"
            f"  Female Acc: {female_acc:.4f}, UAR: {female_rec:.4f}\n"
            f"  Female CM:\n{female_cm}\n"
        )
        print(result)
        with open(results_txt, "a") as f:
            f.write(result + "\n")

    # compute per-fold gender disparities
    acc_diffs = [abs(m - f) for m, f in zip(fold_male_accs, fold_female_accs)]
    uar_diffs = [abs(m - f) for m, f in zip(fold_male_recs, fold_female_recs)]
    # print and log the lists
    print(f"\nFold-wise ACC disparities (male vs female): {acc_diffs}")
    print(f"Fold-wise UAR disparities (male vs female): {uar_diffs}")
    with open(results_txt, "a") as f:
        f.write(f"Fold-wise ACC disparities: {acc_diffs}\n")
        f.write(f"Fold-wise UAR disparities: {uar_diffs}\n")

    # cross-fold summary
    # overall
    avg_acc    = statistics.mean(fold_accs)
    std_acc    = statistics.pstdev(fold_accs)
    avg_rec    = statistics.mean(fold_recs)
    std_rec    = statistics.pstdev(fold_recs)
    # male
    male_avg_acc = statistics.mean(fold_male_accs)
    male_std_acc = statistics.pstdev(fold_male_accs)
    male_avg_rec = statistics.mean(fold_male_recs)
    male_std_rec = statistics.pstdev(fold_male_recs)
    # female
    female_avg_acc = statistics.mean(fold_female_accs)
    female_std_acc = statistics.pstdev(fold_female_accs)
    female_avg_rec = statistics.mean(fold_female_recs)
    female_std_rec = statistics.pstdev(fold_female_recs)

    summary = (
        "\n=== Cross-Fold Summary ===\n"
        f"Overall Acc: {avg_acc:.4f} ± {std_acc:.4f}, UAR: {avg_rec:.4f} ± {std_rec:.4f}\n"
        f"Male    Acc: {male_avg_acc:.4f} ± {male_std_acc:.4f}, UAR: {male_avg_rec:.4f} ± {male_std_rec:.4f}\n"
        f"Female  Acc: {female_avg_acc:.4f} ± {female_std_acc:.4f}, UAR: {female_avg_rec:.4f} ± {female_std_rec:.4f}\n"
    )
    print(summary)
    with open(results_txt, "a") as f:
        f.write(summary)

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