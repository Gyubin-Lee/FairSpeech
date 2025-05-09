import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.RAVDESS_dataset import RAVDESSDataset
from models.feature_extractor import FeatureExtractor
from models.transformer_encoder import TransformerClassifier

from collections import Counter

def collate_fn(batch, max_seq_len, downsample_factor):
    """
    Crop & pad raw waveforms so that after feature‐extractor downsampling
    we have at most `max_seq_len` frames (i.e. max_seq_len*downsample_factor samples).
    Returns:
        waveforms: FloatTensor (batch, max_audio_len)
        emotions : LongTensor (batch,)
        genders  : LongTensor (batch,)
    """
    waveforms, emos, gens = zip(*batch)
    max_audio = max_seq_len * downsample_factor
    cropped = []
    for w in waveforms:
        if w.shape[0] > max_audio:
            w = w[:max_audio]
        cropped.append(w)
    # pad to longest in batch
    lengths = [w.shape[0] for w in cropped]
    max_len = max(lengths)
    batch_size = len(cropped)
    padded = torch.zeros(batch_size, max_len)
    for i, w in enumerate(cropped):
        padded[i, :w.shape[0]] = w
    emos = torch.tensor(emos, dtype=torch.long)
    gens = torch.tensor(gens, dtype=torch.long)
    return padded, emos, gens

def train(cfg):
    # create timestamped output directory
    base_out = Path(cfg['output_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # dump config for reproducibility
    with open(out_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # prepare metrics log
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w") as mf:
        mf.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    device = torch.device(cfg['device'])

    # dataset & dataloader
    train_ds = RAVDESSDataset(
        root=cfg['data_root'],
        split="train",
        sample_rate=cfg.get('sample_rate', 16000)
    )
    val_ds = RAVDESSDataset(
        root=cfg['data_root'],
        split="val",
        sample_rate=cfg.get('sample_rate', 16000)
    )
    down_f = 320

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        collate_fn=lambda b: collate_fn(
            b, cfg['training']['max_seq_len'], down_f
        )
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        collate_fn=lambda b: collate_fn(
            b, cfg['training']['max_seq_len'], down_f
        )
    )

    # feature extractor
    fe = FeatureExtractor(
        model_type=cfg['model']['type'],
        pretrained_model_name_or_path=cfg['model']['pretrained'],
        trainable=cfg['model']['trainable']
    ).to(device)
    feature_dim = fe.model.config.hidden_size

    # classifier
    tconf = cfg['transformer']
    clf = TransformerClassifier(
        feature_dim=feature_dim,
        input_dim=tconf['input_dim'],
        num_layers=tconf['num_layers'],
        nhead=tconf['nhead'],
        dim_feedforward=tconf['dim_feedforward'],
        num_emotions=cfg['training'].get('num_emotions', 7),
        num_genders=cfg['training'].get('num_genders', 2),
        dropout=tconf['dropout'],
        pool=tconf['pool']
    ).to(device)

    # Define loss functions
    emo_counts = Counter([emo for _, emo, _ in train_ds.items])
    num_emotions = len(emo_counts)
    emo_weights = torch.tensor(
        [1.0 / emo_counts[i] for i in range(num_emotions)],
        device=device
    )
    criterion_emo = nn.CrossEntropyLoss(weight=emo_weights)
    criterion_gen = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        [p for p in fe.parameters() if p.requires_grad] + list(clf.parameters()),
        lr=float(cfg['optimizer']['lr']),
        weight_decay=float(cfg['optimizer']['weight_decay'])
    )

    best_val_acc = 0.0

    # training loop
    for epoch in range(1, cfg['training']['epochs'] + 1):
        # train phase
        fe.train() if cfg['model']['trainable'] else fe.eval()
        clf.train()
        running_loss = 0.0
        running_corr = 0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']} [TRAIN]")
        for wave, emos, gens in pbar:
            wave, emos, gens = wave.to(device), emos.to(device), gens.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(cfg['model']['trainable']):
                hidden_states = fe(wave)
            feats = hidden_states[-1]
            emo_logits, gen_logits = clf(feats, lambda_grl=cfg['training']['lambda_grl'])

            loss_emo = criterion_emo(emo_logits, emos)
            loss_gen = criterion_gen(gen_logits, gens)
            loss = loss_emo + cfg['training']['adv_weight'] * loss_gen
            loss.backward()
            optimizer.step()

            bs = wave.size(0)
            running_loss += loss.item() * bs
            preds = emo_logits.argmax(dim=1)
            running_corr += (preds == emos).sum().item()
            running_count += bs

            avg_loss = running_loss / running_count
            avg_acc = running_corr / running_count
            pbar.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{avg_acc:.4f}")

        train_loss = running_loss / running_count
        train_acc = running_corr / running_count

        # validation phase
        fe.eval()
        clf.eval()
        val_loss = 0.0
        val_corr = 0
        val_count = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']} [VAL]  ")
        with torch.no_grad():
            for wave, emos, _ in pbar_val:
                wave, emos = wave.to(device), emos.to(device)
                hidden_states = fe(wave)
                feats = hidden_states[-1]
                emo_logits, _ = clf(feats, lambda_grl=0.0)

                l = criterion_emo(emo_logits, emos)
                bs = wave.size(0)
                val_loss += l.item() * bs
                preds = emo_logits.argmax(dim=1)
                val_corr += (preds == emos).sum().item()
                val_count += bs

                avg_vloss = val_loss / val_count
                avg_vacc = val_corr / val_count
                pbar_val.set_postfix(val_loss=f"{avg_vloss:.4f}", val_acc=f"{avg_vacc:.4f}")

        val_loss = val_loss / val_count
        val_acc = val_corr / val_count

        # log metrics
        with open(metrics_path, "a") as mf:
            mf.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(fe.state_dict(), out_dir / "best_fe.pt")
            torch.save(clf.state_dict(), out_dir / "best_clf.pt")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FairSpeech using a YAML config")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)