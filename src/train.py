#!/usr/bin/env python3
import os
import yaml
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score

from data.RAVDESS_dataset import RAVDESSDataset
from models.feature_extractor import FeatureExtractor
from models.transformer_encoder import TransformerClassifier
from models.pointwise_conv_classifier import PointwiseConv1DClassifier

def collate_fn(batch, max_seq_len, downsample_factor):
    """
    Pads & crops raw waveforms and collects actor_ids in the batch.
    Returns:
        padded: FloatTensor (B, L)
        emos:   LongTensor (B,)
        gens:   LongTensor (B,)
        aids:   LongTensor (B,)
    """
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

def compute_metrics(y_true, y_pred, average='macro'):
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = correct / len(y_true) if y_true else 0.0
    recall = recall_score(y_true, y_pred, average=average) if y_true else 0.0
    return acc, recall

def train(cfg):
    # -- prepare output dir
    base_out = Path(cfg['output_dir'])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # dump config
    with open(out_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # metrics log
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w") as mf:
        mf.write("epoch,train_loss,train_acc,train_recall,val_loss,val_acc,val_recall\n")

    device = torch.device(cfg['device'])

    # -- load speaker stats JSON
    stats_file = Path(cfg['data_root']) / f"speaker_feature_stats_{cfg['model']['type']}.json"
    with open(stats_file) as f:
        raw_stats = json.load(f)
    spk_means = {int(a): torch.tensor(v['mean'], device=device) for a, v in raw_stats.items()}
    spk_stds  = {int(a): torch.tensor(v['std'],  device=device) for a, v in raw_stats.items()}

    # -- datasets & loaders
    train_ds = RAVDESSDataset(root=cfg['data_root'], split="train", sample_rate=cfg.get('sample_rate',16000))
    val_ds   = RAVDESSDataset(root=cfg['data_root'], split="val",   sample_rate=cfg.get('sample_rate',16000))

    down_f = 320
    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['training']['num_workers'],
        collate_fn=lambda b: collate_fn(b, cfg['training']['max_seq_len'], down_f)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['num_workers'],
        collate_fn=lambda b: collate_fn(b, cfg['training']['max_seq_len'], down_f)
    )

    # -- feature extractor
    fe = FeatureExtractor(
        model_type=cfg['model']['type'],
        pretrained_model_name_or_path=cfg['model']['pretrained'],
        trainable=cfg['model']['trainable'],
        use_weighted_sum=cfg['model'].get('use_weighted_sum', False)
    ).to(device)
    fe.eval()  # we won't fine-tune backbone here
    feat_dim = fe.model.config.hidden_size

    # -- classifier selection
    cls_type = cfg['model'].get('classifier', 'transformer').lower()
    if cls_type == 'transformer':
        tconf = cfg['transformer']
        clf = TransformerClassifier(
            feature_dim=feat_dim,
            input_dim=tconf['input_dim'],
            num_layers=tconf['num_layers'],
            nhead=tconf['nhead'],
            dim_feedforward=tconf['dim_feedforward'],
            num_emotions=cfg['training'].get('num_emotions', 7),
            num_genders=cfg['training'].get('num_genders', 2),
            dropout=tconf['dropout'],
            pool=tconf['pool']
        )
    elif cls_type == 'conv1d':
        conv_h = cfg['model'].get('conv_hidden_dim', 128)
        conv_d = cfg['model'].get('conv_dropout', 0.2)
        clf = PointwiseConv1DClassifier(
            input_dim=feat_dim,
            hidden_dim=conv_h,
            num_emotions=cfg['training'].get('num_emotions', 7),
            dropout=conv_d
        )
    else:
        raise ValueError(f"Unknown classifier: {cls_type}")
    clf.to(device)

    # -- loss functions (speaker-norm targets only emo)
    emo_counts = Counter([emo for _, emo, _, _ in train_ds.items])
    weights = torch.tensor([1.0/emo_counts[i] for i in range(len(emo_counts))], device=device)
    criterion_emo = nn.CrossEntropyLoss(weight=weights)
    criterion_gen = nn.CrossEntropyLoss()

    # -- parameter counts and model size
    fe_params = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    clf_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    total_params = fe_params + clf_params
    print(f"Learnable parameters - FeatureExtractor: {fe_params:,}, Classifier: {clf_params:,}, Total: {total_params:,}")
    # Approximate model size in MB (assuming float32, 4 bytes per parameter)
    fe_size_mb = fe_params * 4 / (1024 ** 2)
    clf_size_mb = clf_params * 4 / (1024 ** 2)
    total_size_mb = total_params * 4 / (1024 ** 2)
    print(f"Model size (MB) - FeatureExtractor: {fe_size_mb:.2f} MB, "
          f"Classifier: {clf_size_mb:.2f} MB, Total: {total_size_mb:.2f} MB")

    # -- optimizer & scheduler
    optimizer = torch.optim.AdamW(
        list(clf.parameters()),
        lr=float(cfg['optimizer']['lr']),
        weight_decay=float(cfg['optimizer']['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_recall = 0.0

    # -- training loop
    for epoch in range(1, cfg['training']['epochs']+1):
        # training
        clf.train()
        running_loss = 0.0
        train_trues, train_preds = [], []

        print(f"\nEpoch {epoch}/{cfg['training']['epochs']}  LR: {scheduler.get_last_lr()[0]:.2e}")
        pbar = tqdm(train_loader, desc="TRAIN")
        for wave, emos, gens, aids in pbar:
            wave, emos, gens, aids = wave.to(device), emos.to(device), gens.to(device), aids.to(device)
            optimizer.zero_grad()

            # feature extraction
            out = fe(wave)
            feats = out if isinstance(out, torch.Tensor) else out[-1]  # (B, T, D)

            # optional speaker-wise normalization
            if cfg['model'].get('speaker_wise_normalization', False):
                norm_feats = torch.stack([
                    (feats[i] - spk_means[int(a.item())]) / spk_stds[int(a.item())]
                    for i, a in enumerate(aids)
                ], dim=0)
            else:
                norm_feats = feats

            # forward
            if cls_type == 'transformer':
                emo_logits, gen_logits = clf(norm_feats, lambda_grl=cfg['training']['lambda_grl'])
            else:
                emo_logits = clf(norm_feats)
                gen_logits = None

            # loss
            loss_emo = criterion_emo(emo_logits, emos)
            if gen_logits is not None:
                loss_gen = criterion_gen(gen_logits, gens)
                loss = loss_emo + cfg['training']['adv_weight'] * loss_gen
            else:
                loss = loss_emo

            loss.backward()
            optimizer.step()

            # stats
            bs = wave.size(0)
            running_loss += loss.item() * bs
            preds = emo_logits.argmax(dim=1)
            train_trues.extend(emos.cpu().tolist())
            train_preds.extend(preds.cpu().tolist())

            # inside the TRAIN loop, after updating running_loss, train_trues, train_preds:
            avg_loss = running_loss / len(train_trues)
            avg_acc  = sum(t == p for t, p in zip(train_trues, train_preds)) / len(train_trues)
            pbar.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{avg_acc:.4f}")

        train_loss = running_loss / len(train_trues)
        train_acc, train_recall = compute_metrics(train_trues, train_preds)

        # validation
        clf.eval()
        val_trues, val_preds = [], []
        val_loss = 0.0

        pbar = tqdm(val_loader, desc="VAL  ")
        with torch.no_grad():
            for wave, emos, gens, aids in pbar:
                wave, emos, aids = wave.to(device), emos.to(device), aids.to(device)
                out = fe(wave)
                feats = out if isinstance(out, torch.Tensor) else out[-1]

                if cfg['model'].get('speaker_wise_normalization', False):
                    norm_feats = torch.stack([
                        (feats[i] - spk_means[int(a.item())]) / spk_stds[int(a.item())]
                        for i, a in enumerate(aids)
                    ], dim=0)
                else:
                    norm_feats = feats
                norm_feats = feats
                
                if cls_type == 'transformer':
                    emo_logits, _ = clf(norm_feats, lambda_grl=0.0)
                else:
                    emo_logits = clf(norm_feats)

                l = criterion_emo(emo_logits, emos)
                bs = wave.size(0)
                val_loss += l.item() * bs

                preds = emo_logits.argmax(dim=1)
                val_trues.extend(emos.cpu().tolist())
                val_preds.extend(preds.cpu().tolist())

                # inside the VAL loop:
                avg_vloss = val_loss / len(val_trues)
                avg_vacc  = sum(t == p for t, p in zip(val_trues, val_preds)) / len(val_trues)
                pbar.set_postfix(val_loss=f"{avg_vloss:.4f}", val_acc=f"{avg_vacc:.4f}")

        val_loss = val_loss / len(val_trues)
        val_acc, val_recall = compute_metrics(val_trues, val_preds)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_recall:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_recall:.4f}")

        # log
        with open(metrics_path, "a") as mf:
            mf.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{train_recall:.4f},"
                     f"{val_loss:.4f},{val_acc:.4f},{val_recall:.4f}\n")

        # save best
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(clf.state_dict(), out_dir/"best_clf.pt")

        scheduler.step()

    print(f"\nTraining complete. Best Val Recall: {best_val_recall:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="config.yaml path")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)