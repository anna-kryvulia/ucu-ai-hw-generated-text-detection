"""The fixed validation split made in notebooks/01_eda.ipynb, and the validation metric.

    train = load_split(train)                 # adds group, fold, weight, sub40k; validation = fold 0
    val = train[train["fold"] == 0]
    run.log_metrics("val", scores(val["label"], pred, val["weight"]))

`weight` makes the validation domain mix match the test set; `macro_f1_testlike` uses it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .env import REPO_ROOT

SPLIT_FILE = REPO_ROOT / "results" / "split.csv.gz"
LABELS = ["human", "fusion", "ai"]


def load_split(train: pd.DataFrame, path: str | Path = SPLIT_FILE) -> pd.DataFrame:
    """Adds the columns group, fold, weight and sub40k to train (matched by id)."""
    split = pd.read_csv(path)
    clash = [c for c in split.columns if c != "id" and c in train.columns]
    out = train.drop(columns=clash).merge(split, on="id", how="left", validate="one_to_one")
    assert out["fold"].notna().all(), "some train ids are missing from the split file"
    return out


def scores(y_true, y_pred, weight=None) -> dict:
    """Macro-F1 (plain, and test-like weighted when `weight` is given), accuracy and per-class F1.
    A class that is never predicted gets F1 = 0 (zero_division=0), which is what Kaggle scores too."""
    out = {"macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
           "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))}
    if weight is not None:
        out["macro_f1_testlike"] = f1_score(y_true, y_pred, labels=LABELS, average="macro",
                                            sample_weight=weight, zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out.update({f"f1_{c}": float(v) for c, v in zip(LABELS, per_class)})
    return {k: float(v) for k, v in out.items()}
