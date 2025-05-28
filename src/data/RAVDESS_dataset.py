# src/data/RAVDESS_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa

# emotion label mapping
EMOTION2IDX = {
    'neutral':  0,
    'happy':    1,
    'sad':      2,
    'angry':    3,
    'fearful':  4,
    'disgust':  5,
    'surprise': 6
}
# gender label mapping
GENDER2IDX = {'male': 0, 'female': 1}

class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset for reclassified RAVDESS audio-only files,
    loading with librosa. Now returns (waveform, emotion, gender, actor_id).
    """
    def __init__(self, root: str, split: str = None,
                 sample_rate: int = 16000, transform=None,
                 num_folds: int = None, fold_idx: int = None):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.transform = transform
        self.num_folds = num_folds
        self.fold_idx = fold_idx
        # items: list of tuples (wav_path, emotion_idx, gender_idx, actor_id)
        self.items = []

        # determine actor lists
        all_actors = list(range(1, 21))  # actors 1 through 20
        test_actors = list(range(21, 25))  # fixed test actors
        if num_folds and fold_idx is not None:
            # split the 20 actors into num_folds groups of equal size
            fold_size = len(all_actors) // num_folds
            val_start = (fold_idx - 1) * fold_size
            val_actors = all_actors[val_start:val_start + fold_size]
            train_actors = [a for a in all_actors if a not in val_actors]
        else:
            # original split-based actor filtering
            train_actors = list(range(1, 21)) if split == 'train' else []

        for gender_str, gender_idx in GENDER2IDX.items():
            gender_dir = self.root / gender_str
            if not gender_dir.exists():
                continue
            for actor_dir in sorted(gender_dir.iterdir()):
                if not actor_dir.is_dir():
                    continue
                actor_id = int(actor_dir.name.split("_")[-1])
                # actor-based filtering: either fold-based or original
                if num_folds and fold_idx is not None:
                    if split == 'train' and actor_id not in train_actors:
                        continue
                    if split == 'val' and actor_id not in val_actors:
                        continue
                    if split == 'test' and actor_id not in test_actors:
                        continue
                else:
                    # original split-based
                    if split == 'train' and actor_id not in train_actors:
                        continue
                    if split == 'val'   and actor_id not in val_actors:
                        continue
                    if split == 'test'  and actor_id not in test_actors:
                        continue
                # iterate emotions
                for emo_dir in sorted(actor_dir.iterdir()):
                    emo_str = emo_dir.name
                    if emo_str not in EMOTION2IDX:
                        continue
                    emo_idx = EMOTION2IDX[emo_str]
                    for wav_path in sorted(emo_dir.glob("*.wav")):
                        self.items.append((wav_path, emo_idx, gender_idx, actor_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, emo_idx, gender_idx, actor_id = self.items[idx]
        waveform_np, _ = librosa.load(
            str(wav_path),
            sr=self.sample_rate,
            mono=True
        )
        waveform = torch.from_numpy(waveform_np).float()
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, emo_idx, gender_idx, actor_id

def pad_collate(batch):
    """
    Pads variable-length waveforms to max length in batch.
    Discards actor_id.
    Returns:
        waveforms: FloatTensor (batch, max_len)
        emotions : LongTensor (batch,)
        genders  : LongTensor (batch,)
        lengths  : LongTensor (batch,)  # original lengths
    """
    waveforms, emos, gens, _ = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = lengths.max().item()
    batch_size = len(waveforms)

    padded = torch.zeros(batch_size, max_len)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w

    emos = torch.tensor(emos, dtype=torch.long)
    gens = torch.tensor(gens, dtype=torch.long)
    return padded, emos, gens, lengths

if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Test RAVDESSDataset")
    parser.add_argument("--root", type=str, default="data/RAVDESS_audio/processed",
                        help="Path to processed RAVDESS root")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for DataLoader test")
    args = parser.parse_args()

    print("=== RAVDESS Dataset Sanity Check ===\n")
    for split in ("train", "val", "test"):
        ds = RAVDESSDataset(root=args.root, split=split, sample_rate=16000)
        print(f"Split '{split}': {len(ds)} samples")
        emo_counts = Counter([emo for _, emo, _, _ in ds.items])
        gen_counts = Counter([gen for _, _, gen, _ in ds.items])
        idx2emo = {v: k for k, v in EMOTION2IDX.items()}
        idx2gen = {v: k for k, v in GENDER2IDX.items()}
        print("  Emotion distribution:")
        for idx in sorted(emo_counts):
            print(f"    {idx2emo[idx]:<8}: {emo_counts[idx]}")
        print("  Gender distribution:")
        for idx in sorted(gen_counts):
            print(f"    {idx2gen[idx]:<6}: {gen_counts[idx]}")
        print()

    print("=== DataLoader Test ===")
    ds = RAVDESSDataset(root=args.root, split="train", sample_rate=16000)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, collate_fn=pad_collate, num_workers=0)

    wave_batch, emo_batch, gen_batch, lengths = next(iter(loader))
    print(f"waveforms: {wave_batch.shape}")
    print(f"emotions:  {emo_batch}")
    print(f"genders:   {gen_batch}")
    print(f"lengths:   {lengths.tolist()}")