import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def split_gtsrb_val(root="data", val_percent=0.2, seed=42, copy=True):
    random.seed(seed)
    root = Path(root)
    train_dir = root / "train" / "GTSRB"
    val_dir = root / "val" / "GTSRB"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    val_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 Splitting GTSRB data → {val_dir} ({val_percent*100:.1f}% per class)")

    for class_dir in tqdm(sorted(train_dir.iterdir()), desc="Processing classes"):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        val_class_dir = val_dir / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))
        if len(images) == 0:
            continue

        n_val = max(1, int(len(images) * val_percent))
        val_samples = random.sample(images, n_val)

        for img_path in val_samples:
            dst = val_class_dir / img_path.name
            if copy:
                shutil.copy2(img_path, dst)
            else:
                shutil.move(img_path, dst)

        print(f"{class_name}: {len(images)} total → {n_val} moved to val")

    print("\n Split complete!")
    print(f"Train data kept in: {train_dir}")
    print(f"Validation data in: {val_dir}")
    print(f"Mode: {'copy' if copy else 'move'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split GTSRB dataset into train/val subsets per class.")
    parser.add_argument("--root", type=str, default="data",
                        help="Root dataset folder containing train/GTSRB/")
    parser.add_argument("--val-percent", type=float, default=0.2,
                        help="Fraction of images per class to use for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--move", action="store_true", help="Move instead of copy")
    args = parser.parse_args()

    split_gtsrb_val(root=args.root, val_percent=args.val_percent, seed=args.seed, copy=not args.move)
