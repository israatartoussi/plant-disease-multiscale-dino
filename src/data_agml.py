# src/data_agml.py  — organize-only (no agml import)
import os, json, random, shutil, argparse, yaml
from pathlib import Path
from datetime import date
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

AGML_DIRMAP = {
    "corn_maize_leaf_disease":   "corn_maize_leaf_disease",            # ← صح
    "bean_disease_uganda":       "bean_disease_uganda",
    "guava_disease_pakistan":    "guava_disease_pakistan",
    "papaya_leaf_disease":       "papaya_leaf_disease_classification",
    "blackgram_leaf_disease":    "blackgram_plant_leaf_disease_classification",
    "banana_leaf_disease":       "banana_leaf_disease_classification",
    "coconut_tree_disease":      "coconut_tree_disease_classification",
    "rice_leaf_disease":         "rice_seedling_segmentation",         # منستعمل هيدا كبديل للـ rice
    "sunflower_disease":         "sunflower_disease_classification",
}

def agml_base():
    return Path.home() / ".agml" / "datasets"

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def deterministic_split(files, ratios, seed):
    files = list(files)
    rnd = random.Random(seed); rnd.shuffle(files)
    n = len(files); n_tr = int(ratios["train"]*n); n_va = int(ratios["val"]*n)
    return files[:n_tr], files[n_tr:n_tr+n_va], files[n_tr+n_va:]

def copy_pairs(pairs, out_root: Path):
    for cls, src in pairs:
        dst = out_root / cls / src.name
        if not dst.exists():
            safe_mkdir(dst.parent); shutil.copy2(src, dst)

def collect_by_class(raw_root: Path):
    # نفترض هيكلة تصنيف: .../<class>/<image>
    cls_to_files = defaultdict(list)
    for img in list_images(raw_root):
        cls = img.parent.name
        cls_to_files[cls].append(img)
    return cls_to_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg   = yaml.safe_load(open(args.config, "r"))
    names = cfg["names"]; split = cfg["split"]; seed = int(cfg.get("seed", 42))
    out_root = Path(cfg.get("root", "data"))
    base = agml_base(); print("AgML dir:", base)

    for name in names:
        print(f"\n==> Dataset: {name}")
        raw_name = AGML_DIRMAP.get(name, name)
        raw_dir = base / raw_name
        if not raw_dir.exists():
            print(f"[ERROR] not found: {raw_dir}")
            print("   → افتحي ناتج ls ~/.agml/datasets وحطي الاسم الصحيح بـ AGML_DIRMAP.")
            continue

        cls_to_files = collect_by_class(raw_dir)
        if not cls_to_files:
            print(f"[ERROR] no class subfolders under: {raw_dir}")
            continue

        ds_dir = out_root / name
        for sp in ["train","val","test"]: safe_mkdir(ds_dir / sp)

        pairs_tr = []; pairs_va = []; pairs_te = []
        classes = sorted(cls_to_files.keys()); total = 0
        for c, files in cls_to_files.items():
            tr, va, te = deterministic_split(files, split, seed)
            pairs_tr += [(c, p) for p in tr]
            pairs_va += [(c, p) for p in va]
            pairs_te += [(c, p) for p in te]
            total += len(files)

        copy_pairs(pairs_tr, ds_dir / "train")
        copy_pairs(pairs_va, ds_dir / "val")
        copy_pairs(pairs_te, ds_dir / "test")

        (ds_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        manifest = {
            "source": "agml",
            "agml_dir": str(raw_dir),
            "download_date": str(date.today()),
            "seed": seed,
            "split": split,
            "classes": classes,
            "total_files_found": total,
        }
        (ds_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] {name}: classes={len(classes)} | files={total} | out={ds_dir}")

if __name__ == "__main__":
    main()
