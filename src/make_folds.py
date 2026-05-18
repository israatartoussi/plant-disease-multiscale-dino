import json, argparse
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold

IMG_EXT = {'.jpg','.jpeg','.png','.bmp','.gif','.tif','.tiff','.webp'}

def load_annotations(dataset: str) -> Tuple[List[str], List[int]]:
    """
    يحاول أولاً قراءة data/<dataset>/annotations.json بالشكل:
      [{"img": "path/to/img", "label": int}, ...]
    إذا الملف غير موجود، يحاول بناء الannotations بمسح مجلدات الكلاسات:
      data/<dataset>/<class_name>/**/<image files>
    ويعطي label = ترتيب اسم الكلاس أبجديًا.
    """
    ann_path = Path(f"data/{dataset}/annotations.json")
    if ann_path.exists():
        ann = json.load(open(ann_path))
        X = [a["img"] for a in ann]
        y = [int(a["label"]) for a in ann]
        return X, y

    root = Path(f"data/{dataset}")
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    # حاول استنتاج مجلدات الكلاسات من تحت جذر الداتا
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    blacklist = {'annotations','splits','folds','images','img','train','val','test'}
    class_dirs = [p for p in class_dirs if p.name.lower() not in blacklist]

    # لو في بنية train/val/test، خذ train لتحديد الكلاسات
    tvt = [p for p in [root/'train', root/'val', root/'test'] if p.exists()]
    if tvt:
        train_dir = root/'train'
        if train_dir.exists():
            class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]

    class_names = sorted([p.name for p in class_dirs])
    if not class_names:
        # fallback: ابحث تحت images/
        maybe = root/'images'
        if maybe.exists():
            class_dirs = [p for p in maybe.iterdir() if p.is_dir()]
            class_names = sorted([p.name for p in class_dirs])

    if not class_names:
        raise RuntimeError(f"Could not infer class folders under {root}")

    class_to_idx = {c:i for i,c in enumerate(class_names)}
    X, y = [], []

    def is_img(p: Path) -> bool:
        return p.suffix.lower() in IMG_EXT

    def collect_from(base: Path, c: str):
        cdir = base / c
        if not cdir.exists():
            return False
        for p in cdir.rglob('*'):
            if p.is_file() and is_img(p):
                X.append(str(p.as_posix()))
                y.append(class_to_idx[c])
        return True

    for c in class_names:
        # جرّب تحت الجذر مباشرة
        if collect_from(root, c):
            continue
        # جرّب تحت images/ ثم train/ ثم val/ ثم test/
        for sub in ['images','train','val','test']:
            if collect_from(root/sub, c):
                break

    if not X:
        raise RuntimeError(f"No images found under {root}. Check dataset structure.")

    # خزّن annotations.json لتسهيل الاستخدام لاحقًا
    ann_out = [{'img': img, 'label': int(lbl)} for img,lbl in zip(X,y)]
    out_ann = Path(f"data/{dataset}/annotations.json")
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ann_out, open(out_ann,'w'))
    print(f"[make_folds] wrote {out_ann} (auto-generated)")

    return X, y

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    X, y = load_annotations(args.dataset)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    folds = {f"fold_{k}": {"train_idx": [], "val_idx": []} for k in range(args.n_folds)}
    for k, (tr, va) in enumerate(skf.split(X, y)):
        folds[f"fold_{k}"]["train_idx"] = tr.tolist()
        folds[f"fold_{k}"]["val_idx"] = va.tolist()

    outp = Path(f"data/{args.dataset}/folds_{args.n_folds}.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(folds, open(outp, "w"))
    print(f"[make_folds] wrote {outp}")
