#!/usr/bin/env python3
import shutil
from pathlib import Path
import argparse

# Map RAVDESS emotion codes to merged classes (neutral + calm)
EMOTION_MAP = {
    '01': 'neutral',   # neutral
    '02': 'neutral',   # calm → merged into neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprise'
}

def reclassify_ravdess(raw_root: Path, processed_root: Path):
    """
    Reclassify RAVDESS audio-only speech files into:
      processed_root/
        ├── male/
        │    └── Actor_01/
        │         └── <emotion>/
        └── female/
             └── Actor_02/
                  └── <emotion>/
    Neutral and calm are merged into 'neutral'.
    """
    for wav_path in raw_root.rglob('*.wav'):
        parts = wav_path.stem.split('-')
        if len(parts) != 7:
            continue
        modality, channel, emo_code, intensity, stmt, rep, actor = parts

        # map emotion code to label
        emotion = EMOTION_MAP.get(emo_code)
        if emotion is None:
            continue

        # determine gender by actor ID (odd=male, even=female)
        actor_id = int(actor)
        gender = 'male' if actor_id % 2 == 1 else 'female'
        actor_folder = f"Actor_{actor}"

        # build destination directory
        dest_dir = processed_root / gender / actor_folder / emotion
        dest_dir.mkdir(parents=True, exist_ok=True)

        # copy the .wav file
        dest_file = dest_dir / wav_path.name
        shutil.copy2(wav_path, dest_file)
        print(f"Copied {wav_path.name} → {gender}/{actor_folder}/{emotion}/")

def main():
    parser = argparse.ArgumentParser(
        description="Reclassify RAVDESS audio-only speech and song files"
    )
    parser.add_argument(
        "--raw_root", type=Path, default=Path("raw"),
        help="Path to raw RAVDESS root (contains Actor_XX subfolders)"
    )
    parser.add_argument(
        "--processed_root", type=Path, default=Path("processed"),
        help="Path to output reclassified dataset"
    )
    args = parser.parse_args()
    reclassify_ravdess(args.raw_root, args.processed_root)

if __name__ == "__main__":
    main()