import argparse
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def gather_samples(ds_root: Path, split_mode: str):
    if split_mode == "train_only":
        split_dirs = [ds_root / "train"]
    else:
        split_dirs = [ds_root / "train", ds_root / "val", ds_root / "test"]

    samples = []
    for sdir in split_dirs:
        if not sdir.exists():
            continue
        for cdir in sorted([d for d in sdir.iterdir() if d.is_dir()]):
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXT:
                    samples.append((p, cdir.name))
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_mode", choices=["train_only", "all"], default="train_only")
    args = ap.parse_args()

    ds_root = Path(args.data_root) / args.dataset
    if not ds_root.exists():
        print(f"[ERROR] dataset folder not found: {ds_root}")
        return

    samples = gather_samples(ds_root, args.split_mode)
    if len(samples) == 0:
        print(f"[ERROR] No images found under: {ds_root} (split_mode={args.split_mode})")
        print("Expected: data/<ds>/(train|val|test)/<class>/*.(jpg/png/...)")
        return

    paths, labels_str = zip(*samples)
    classes = sorted(set(labels_str))
    cls2id = {c: i for i, c in enumerate(classes)}
    y = [cls2id[c] for c in labels_str]

    print("== Inspect Folds ==")
    print(f"Dataset: {args.dataset}")
    print(f"Root: {ds_root}")
    print(f"Split mode: {args.split_mode}")
    print(f"Total images: {len(samples)}")
    print(f"Classes ({len(classes)}): {cls2id}")
    print()

    total_counts = Counter(labels_str)
    print("[TOTAL COUNTS]")
    for c in classes:
        print(f"  {c:25s}: {total_counts[c]}")
    print()

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    for fold, (_, val_idx) in enumerate(skf.split(range(len(y)), y)):
        fold_counts = Counter([labels_str[i] for i in val_idx])
        print(f"--- Fold {fold} (val size = {len(val_idx)}) ---")
        missing = []
        for c in classes:
            n = fold_counts.get(c, 0)
            print(f"  {c:25s}: {n}")
            if n == 0:
                missing.append(c)
        if missing:
            print(f"  [WARN] Missing classes in this fold: {missing}")
        print()

    print("[DONE]")

if __name__ == "__main__":
    main()
