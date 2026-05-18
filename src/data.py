# src/data.py
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_transforms(img_size: int = 224):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tfm_train, tfm_eval

def build_loaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    data_root = Path(data_root)
    tfm_train, tfm_eval = build_transforms(img_size)

    ds_train = datasets.ImageFolder((data_root / "train").as_posix(), transform=tfm_train)
    ds_val   = datasets.ImageFolder((data_root / "val").as_posix(),   transform=tfm_eval)
    ds_test  = datasets.ImageFolder((data_root / "test").as_posix(),  transform=tfm_eval)

    tr = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    va = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    te = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)


    classes = ds_train.classes
    return tr, va, te, classes
