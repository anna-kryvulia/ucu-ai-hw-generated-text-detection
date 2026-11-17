"""Kaggle submission files: same ids and columns as sample_submission.csv, checked before saving.

    make_submission(test_df["id"], predicted_labels, run.dir / "submission.csv", P.data)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

LABELS = ("ai", "fusion", "human")


def make_submission(ids: Iterable, labels: Iterable, out_path: str | Path, data_root: str | Path) -> pd.DataFrame:
    sample = pd.read_csv(Path(data_root) / "sample_submission.csv")
    sub = pd.DataFrame({"id": list(ids), sample.columns[1]: list(labels)})
    check_submission(sub, sample)
    sub = sub.set_index("id").loc[sample["id"]].reset_index()     # same row order as the sample
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"submission saved: {out_path}  ({len(sub)} rows)")
    print(sub[sample.columns[1]].value_counts(normalize=True).round(3).to_string())
    return sub


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert sub["id"].is_unique, "duplicate ids"
    missing = set(sample["id"]) - set(sub["id"])
    extra = set(sub["id"]) - set(sample["id"])
    assert not missing and not extra, f"{len(missing)} ids missing, {len(extra)} unknown ids"
    bad = set(sub[sample.columns[1]]) - set(LABELS)
    assert not bad, f"unknown labels {bad}; allowed: {LABELS}"
