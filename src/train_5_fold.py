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
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

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

    # define actor splits for 5-fold CV (actors 1–20)
    val_actor_ranges = [
        list(range(17, 21)),  # fold1: 17-20
        list(range(1, 5)),    # fold2: 1-4
        list(range(5, 9)),    # fold3: 5-8
        list(range(9, 13)),   # fold4: 9-12
        list(range(13, 17))   # fold5: 13-16
    ]

    # load fixed test set if needed (not used here)
    # test_ds = RAVDESSDataset(root=..., split="test", ...)

    for fold_idx, val_actors in enumerate(val_actor_ranges, start=1):
        print(f"\n=== Fold {fold_idx} validation actors: {val_actors} ===")
        # create out_dir per fold
        fold_out = out_dir / f"fold_{fold_idx}"
        fold_out.mkdir(parents=True, exist_ok=True)
        # dump config per fold
        with open(fold_out/"used_config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
        # metrics CSV per fold
        mpath = fold_out/"metrics.csv"
        with open(mpath, "w") as mf:
            mf.write(
                "epoch,train_loss,train_acc,train_recall,"
                "val_loss,val_acc,val_recall,"
                "male_loss,male_acc,male_recall,"
                "female_loss,female_acc,female_recall\n"
            )
        # fold-aware datasets
        train_ds = RAVDESSDataset(
            root=cfg['data']['root'],
            split="train",
            sample_rate=cfg.get('sample_rate', 16000),
            num_folds=len(val_actor_ranges),
            fold_idx=fold_idx
        )
        val_ds = RAVDESSDataset(
            root=cfg['data']['root'],
            split="val",
            sample_rate=cfg.get('sample_rate', 16000),
            num_folds=len(val_actor_ranges),
            fold_idx=fold_idx
        )

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
        ).to(cfg['device'])
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
                pool=tconf['pool'],
                speaker_wise_normalization=cfg['model'].get('speaker_wise_normalization', False),
                predict_gender=tconf.get('predict_gender', False)
            )
        elif cls_type == 'conv1d':
            conv_h = cfg['conv1d']['hidden_dim']
            conv_d = cfg['conv1d']['dropout']
            clf = PointwiseConv1DClassifier(
                input_dim=feat_dim,
                hidden_dim=conv_h,
                num_emotions=cfg['training']['num_emotions'],
                dropout=conv_d,
                speaker_wise_normalization=cfg['model'].get('speaker_wise_normalization', False)
            )
        else:
            raise ValueError(f"Unknown classifier: {cls_type}")
        clf.to(cfg['device'])

        device = torch.device(cfg['device'])

        # fairness loss option
        use_l2_loss = cfg['training'].get('use_l2_loss', False)
        fairness_weight = float(cfg['training'].get('fairness_weight', 1.0))

        # -- loss functions (speaker-norm targets only emo)
        emo_counts = Counter([emo for _, emo, _, _ in train_ds.items])
        total_samples = sum(emo_counts.values())
        num_classes = len(emo_counts)
        weights = torch.tensor(
            [ total_samples / (num_classes * emo_counts[i]) for i in range(num_classes) ],
            device=device
        )
        criterion_emo = nn.CrossEntropyLoss(weight=weights)
        criterion_emo_none = nn.CrossEntropyLoss(weight=weights, reduction='none')
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
        # determine learning rate
        if cls_type == 'transformer':
            lr = float(cfg['transformer']['lr'])
        else:
            lr = float(cfg['conv1d']['lr'])
        
        # include any trainable feature-extractor parameters (e.g., layer_weights)
        fe_trainable = [p for p in fe.parameters() if p.requires_grad]
        clf_trainable = [p for p in clf.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            fe_trainable + clf_trainable,
            lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        best_val_loss = 10.0
        no_improve_count = 0
        patience = 10

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

                # forward
                if cls_type == 'transformer':
                    emo_logits, gen_logits = clf(feats, lambda_grl=cfg['training']['lambda_grl'])
                else:
                    emo_logits = clf(feats)
                    gen_logits = None

                # emotion loss
                loss_emo = criterion_emo(emo_logits, emos)

                if use_l2_loss:
                    # per-sample CE losses
                    per_sample = cross_entropy(emo_logits, emos, reduction='none')
                    # masks
                    mask_m = gens == 0
                    mask_f = gens == 1
                    if mask_m.any() and mask_f.any():
                        Lm = per_sample[mask_m].mean()
                        Lf = per_sample[mask_f].mean()
                        adv_loss = (Lm - Lf) ** 2
                    else:
                        adv_loss = torch.tensor(0.0, device=device)
                    loss = loss_emo + fairness_weight * adv_loss
                elif gen_logits is not None:
                    # original adversarial term
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
                pbar.set_postfix(train_loss=f"{avg_loss:.4f}", train_acc=f"{avg_acc:.4f}",
                                 adv_loss=f"{adv_loss.item():.4f}" if use_l2_loss else "")

            train_loss = running_loss / len(train_trues)
            train_acc, train_recall = compute_metrics(train_trues, train_preds)

            # validation
            clf.eval()
            # per-gender accumulators
            val_trues_g = {0: [], 1: []}
            val_preds_g = {0: [], 1: []}
            val_loss_g  = {0: 0.0, 1: 0.0}
            val_count_g = {0: 0,   1: 0}
            val_trues, val_preds = [], []
            val_loss = 0.0

            pbar = tqdm(val_loader, desc="VAL  ")
            with torch.no_grad():
                for wave, emos, gens, aids in pbar:
                    wave, emos, gens, aids = wave.to(device), emos.to(device), gens.to(device), aids.to(device)
                    out = fe(wave)
                    feats = out if isinstance(out, torch.Tensor) else out[-1]

                    if cls_type == 'transformer':
                        emo_logits, _ = clf(feats, lambda_grl=0.0)
                    else:
                        emo_logits = clf(feats)

                    # per-sample losses for gender grouping
                    losses = criterion_emo_none(emo_logits, emos)  # (bs,)
                    for loss_i, g in zip(losses, gens):
                        grp = int(g.item())
                        val_loss_g[grp]  += loss_i.item()
                        val_count_g[grp] += 1
                    val_loss += losses.sum().item()

                    preds = emo_logits.argmax(dim=1)
                    # collect overall
                    val_trues.extend(emos.cpu().tolist())
                    val_preds.extend(preds.cpu().tolist())
                    # collect per-gender
                    for emo_t, pred, g in zip(emos.cpu().tolist(), preds.cpu().tolist(), gens.cpu().tolist()):
                        val_trues_g[g].append(emo_t)
                        val_preds_g[g].append(pred)

                    # inside the VAL loop:
                    avg_vloss = val_loss / len(val_trues)
                    avg_vacc  = sum(t == p for t, p in zip(val_trues, val_preds)) / len(val_trues)
                    pbar.set_postfix(val_loss=f"{avg_vloss:.4f}", val_acc=f"{avg_vacc:.4f}")

            val_loss = val_loss / len(val_trues)
            val_acc, val_recall = compute_metrics(val_trues, val_preds)

            # per-gender metrics
            male_loss = val_loss_g[0] / val_count_g[0] if val_count_g[0] > 0 else 0.0
            female_loss = val_loss_g[1] / val_count_g[1] if val_count_g[1] > 0 else 0.0
            male_acc, male_recall   = compute_metrics(val_trues_g[0], val_preds_g[0])
            female_acc, female_recall = compute_metrics(val_trues_g[1], val_preds_g[1])

            print(f"\nEpoch {epoch} summary:")
            print(f"  Train loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall: {train_recall:.4f}")
            print(f"  Val   loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall: {val_recall:.4f}")
            print(f"    Male   loss: {male_loss:.4f}, Acc: {male_acc:.4f}, Recall: {male_recall:.4f}")
            print(f"    Female loss: {female_loss:.4f}, Acc: {female_acc:.4f}, Recall: {female_recall:.4f}")

            # log
            with open(mpath, "a") as mf:
                mf.write(
                    f"{epoch},{train_loss:.4f},{train_acc:.4f},{train_recall:.4f},"
                    f"{val_loss:.4f},{val_acc:.4f},{val_recall:.4f},"
                    f"{male_loss:.4f},{male_acc:.4f},{male_recall:.4f},"
                    f"{female_loss:.4f},{female_acc:.4f},{female_recall:.4f}\n"
                )

            # early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
                # save both feature extractor and classifier
                torch.save(fe.state_dict(), fold_out/"best_fe.pt")
                torch.save(clf.state_dict(), fold_out/"best_clf.pt")
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"No improvement in val_recall for {patience} epochs, stopping early.")
                    break

            scheduler.step()


        print(f"\nTraining complete. Best Val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="config.yaml path")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)