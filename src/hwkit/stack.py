"""Features and cross-validation helpers for the stacker that is fitted on the validation fold.

    folds = group_folds(val)                          # 5 group folds inside validation
    X = pd.concat([chunk_features(npz, n, "xlmr"), text_features(val["text"])], axis=1)
    oof = out_of_fold(fit, apply, folds)              # every row predicted by a model that did not see it
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .split import LABELS


def group_folds(val: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Fold number of every validation row; source groups stay together (the same folds as notebook 04)."""
    folds = np.zeros(len(val), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(sgkf.split(val, val["label"], val["group"])):
        folds[idx] = k
    return folds


def out_of_fold(fit: Callable, apply: Callable, folds: np.ndarray, n_classes: int = len(LABELS)) -> np.ndarray:
    """fit(train_mask) -> model; apply(model, held_out_mask) -> class scores of the held-out rows."""
    oof = np.zeros((len(folds), n_classes))
    for k in np.unique(folds):
        oof[folds == k] = apply(fit(folds != k), folds == k)
    return oof


def macro_f1(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    """Weighted macro-F1 over the classes 0..2 (same value as sklearn, faster inside grid searches)."""
    f1 = []
    for c in range(len(LABELS)):
        tp = weight[(y == c) & (pred == c)].sum()
        wrong = weight[(y == c) != (pred == c)].sum()            # false positives + false negatives
        f1.append(2 * tp / (2 * tp + wrong) if tp + wrong > 0 else 0.0)
    return float(np.mean(f1))


def chunk_features(npz: dict, n_texts: int, prefix: str) -> pd.DataFrame:
    """Per text, from the class probabilities of its chunks: log of the mean and of the first chunk, and the shape
    across chunks (highest ai and fusion, lowest and highest human, spread of human). A fusion text is a human text
    with one AI-written section, so a single chunk can differ from the rest."""
    doc, pos, probs = npz["doc"], npz["pos"], npz["probs"]
    df = pd.DataFrame(probs, columns=LABELS).assign(doc=doc, pos=pos)
    g = df.groupby("doc")
    out = pd.DataFrame(index=pd.RangeIndex(n_texts))
    for c in LABELS:
        out[f"{prefix}_mean_{c}"] = np.log(np.clip(g[c].mean().reindex(out.index).to_numpy(), 1e-6, 1))
        out[f"{prefix}_first_{c}"] = np.log(np.clip(df[df["pos"] == 0].set_index("doc")[c].reindex(out.index)
                                                    .to_numpy(), 1e-6, 1))
    out[f"{prefix}_max_ai"] = g["ai"].max().reindex(out.index).to_numpy()
    out[f"{prefix}_max_fusion"] = g["fusion"].max().reindex(out.index).to_numpy()
    out[f"{prefix}_min_human"] = g["human"].min().reindex(out.index).to_numpy()
    out[f"{prefix}_max_human"] = g["human"].max().reindex(out.index).to_numpy()
    out[f"{prefix}_std_human"] = g["human"].std(ddof=0).reindex(out.index).to_numpy()
    out[f"{prefix}_log_chunks"] = np.log(g.size().reindex(out.index).to_numpy())
    return out


SEAMS = {   # formatting traces that can appear where generated text was pasted into a human text (per 100 words)
    "space_before_closing_quote": r"[.!?,] [\"”'](?: |$)",
    "space_before_punct": r"\w [,.;:!?](?: |$)",
    "double_space": r"  ",
    "no_space_after_sentence": r"[a-z][.!?][A-Z][a-z]",
}


def text_features(texts: pd.Series) -> pd.DataFrame:
    """Seam traces and a few style statistics of each text."""
    t = texts.reset_index(drop=True)
    words = t.str.split()
    n = words.str.len().clip(lower=1)
    out = pd.DataFrame({f"seam_{k}": 100 * t.str.count(p) / n for k, p in SEAMS.items()})
    out["log_words"] = np.log(n)
    out["mean_word_len"] = t.str.len() / n
    out["type_token_200"] = words.map(lambda w: len(set(w[:200])) / max(len(w[:200]), 1))   # same window for all
    out["digits_per_100w"] = 100 * t.str.count(r"\d") / n
    out["quotes_per_100w"] = 100 * t.str.count(r"[\"“”„«»]") / n
    out["commas_per_100w"] = 100 * t.str.count(",") / n
    return out

