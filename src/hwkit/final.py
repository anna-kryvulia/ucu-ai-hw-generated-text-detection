"""Feature blocks of the final stacker (notebook 13), as functions, so that notebook 14 can rebuild exactly the same
features to explain the model. Notebook 13 has the same code inline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hwkit.split import LABELS
from hwkit.stack import chunk_features, text_features
from hwkit.tagger import load_tokens, tag_features

OLD_TAG = ["tag_mean", "tag_share", "tag_std", "tag_runs", "tag_longest_share", "tag_longest_log",
           "tag_smooth_max", "tag_smooth_min"]                     # the eight features of notebook 10
FINAL_BLOCKS = ["base", "shape", "text", "tagger", "fine1", "avg"]   # feature set of the final model


def run_features(runs: Path, run_id: str, split: str, ids: pd.Series) -> pd.DataFrame:
    """Chunk features of a saved encoder run, rows in the order of `ids`."""
    pred_dir = runs / run_id / "predictions"
    order = pd.read_csv(pred_dir / f"{split}.csv", usecols=["id"])["id"]
    f = chunk_features(dict(np.load(pred_dir / f"chunks_{split}.npz")), len(order), run_id)
    return f.set_index(order).loc[ids].reset_index(drop=True)


def averaged_tokens(token_dir: Path, split: str):
    """Token-wise average of both taggers' AI probabilities: (ids, list of probability arrays, list of starts)."""
    ids, p1, starts = load_tokens(token_dir / f"tokens_tagger1_{split}.npz")
    ids2, p2, _ = load_tokens(token_dir / f"tokens_{split}.npz")
    assert (ids == ids2).all() and all(len(a) == len(b) for a, b in zip(p1, p2))
    return ids, [(a + b) / 2 for a, b in zip(p1, p2)], starts


def _summaries(ids, probs, starts, df: pd.DataFrame, windows) -> pd.DataFrame:
    text_of = df.set_index("id")["text"]
    f = pd.DataFrame([tag_features(p, s, text_of[i], windows, fine=True) for i, p, s in zip(ids, probs, starts)],
                     index=ids)
    return f.loc[df["id"]].reset_index(drop=True)


def feature_blocks(runs: Path, cfg: dict, split: str, df: pd.DataFrame, ohe) -> dict[str, pd.DataFrame]:
    """All feature blocks of notebook 13 for one split; `ohe` is the domain encoder fitted on validation."""
    tagger = pd.read_csv(runs / cfg["inputs"]["tagger"] / "predictions" / f"features_{split}.csv").set_index("id")
    tfidf = pd.read_csv(runs / cfg["inputs"]["baseline"] / "predictions" / f"{split}.csv").set_index("id")
    enc = pd.concat([run_features(runs, r, split, df["id"]) for r in cfg["inputs"]["encoders"]], axis=1)
    base_cols = [c for c in enc.columns if "_mean_" in c or "_first_" in c or c.endswith("_log_chunks")]
    token_dir = runs / cfg["inputs"]["tagger_tokens"] / "predictions"
    windows = cfg["features"]["windows"]
    first = _summaries(*load_tokens(token_dir / f"tokens_tagger1_{split}.npz"), df, windows)
    return {
        "base": pd.concat([enc[base_cols],
                           np.log(np.clip(tfidf.loc[df["id"], [f"s_{c}" for c in LABELS]], 1e-6, 1))
                           .reset_index(drop=True).add_prefix("tfidf_"),
                           pd.DataFrame(ohe.transform(df[["domain_lang"]]), columns=ohe.categories_[0])], axis=1),
        "shape": enc.drop(columns=base_cols),
        "text": text_features(df["text"]),
        "tagger": tagger.loc[df["id"]].reset_index(drop=True),
        "fine1": first.drop(columns=OLD_TAG),
        "avg": _summaries(*averaged_tokens(token_dir, split), df, windows).add_prefix("avg_"),
    }
