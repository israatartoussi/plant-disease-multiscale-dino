# src/utils_dino.py
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def prf1_from_predictions(y_true, y_pred):
    """
    Compute classification metrics from full-set predictions.
    y_true, y_pred: numpy arrays or lists
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "f1_weighted": float(f1w),
    }


@torch.no_grad()
def prf1_macro_weighted(logits, y):
    """
    Batch-level helper only.
    Prefer full-set metrics for validation/test reporting.
    """
    preds = logits.argmax(1).detach().cpu().numpy()
    ytrue = y.detach().cpu().numpy()
    return prf1_from_predictions(ytrue, preds)


@torch.no_grad()
def fullset_classification_metrics(y_true, y_pred, ncls=None):
    """
    Full evaluation over the whole dataset.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = prf1_from_predictions(y_true, y_pred)
    metrics["acc"] = float((y_true == y_pred).mean())

    if ncls is None:
        labels = None
    else:
        labels = np.arange(ncls)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    metrics["cm"] = cm.tolist()
    metrics["cm_norm"] = cm_norm.tolist()

    return metrics


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_ckpt(path: Path, model, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)