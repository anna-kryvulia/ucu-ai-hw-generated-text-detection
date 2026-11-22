"""Token tagger helpers: token labels, 512-token chunks, prediction, token-probability files and per-text features.

Notebook 10 has the same code inline; notebooks 13 and 14 use this module. A token tagger predicts, for every token of
a text, the probability that an AI wrote it. Long texts are cut into chunks of `body` tokens (512 minus the start and
end tokens); every token gets the probability from the chunk that contains it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from hwkit.spans import parse_spans

IGNORE = -100          # label of start/end and padding tokens; the loss skips it
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"”'»)]*\s+")


def token_labels(starts: np.ndarray, label: str, spans) -> np.ndarray | None:
    """0 = human, 1 = AI for every token (by its first character); None for a fusion text without known sections."""
    if label == "human":
        return np.zeros(len(starts), dtype=np.int8)
    if label == "ai":
        return np.ones(len(starts), dtype=np.int8)
    if not isinstance(spans, str):
        return None
    lab = np.zeros(len(starts), dtype=np.int8)
    for s, e in parse_spans(spans):
        lab[(starts >= s) & (starts < e)] = 1
    return lab


def tokenize(tok, df: pd.DataFrame, span_of: pd.Series | None = None, batch: int = 2000):
    """Token ids (int32), token labels (int8, or None) and first-character positions (int32) of every text."""
    ids, labs, starts = [], [], []
    for i in range(0, len(df), batch):
        part = df.iloc[i:i + batch]
        enc = tok(part["text"].tolist(), add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        for k, (x, off) in enumerate(zip(enc["input_ids"], enc["offset_mapping"])):
            st = np.fromiter((o[0] for o in off), dtype=np.int32, count=len(off))
            ids.append(np.asarray(x, dtype=np.int32))
            starts.append(st)
            labs.append(token_labels(st, part["label"].iat[k], span_of.get(part["id"].iat[k]))
                        if span_of is not None and "label" in part else None)
    return ids, labs, starts


def chunk_starts(n: int, body: int) -> list[int]:
    """Start positions of the chunks covering n tokens; the last chunk ends exactly at the end of the text."""
    if n <= body:
        return [0]
    return list(range(0, n - body, body)) + [n - body]


class Chunks:
    """Turns token lists into model inputs for one tokenizer and chunk length."""

    def __init__(self, tok, max_len: int):
        self.tok, self.body = tok, max_len - 2

    def wrap(self, x):
        return np.concatenate([[self.tok.cls_token_id], x, [self.tok.sep_token_id]]).astype(np.int64)

    @staticmethod
    def wrap_labels(lab):
        return np.concatenate([[IGNORE], lab, [IGNORE]]).astype(np.int64)

    def collate(self, batch):
        """Pad to the longest chunk of the batch; padding gets the IGNORE label and attention mask 0."""
        xs, ys = zip(*batch)
        width = max(len(x) for x in xs)
        input_ids = torch.full((len(xs), width), self.tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(xs), width), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(xs), width), dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            input_ids[i, :len(x)] = torch.from_numpy(x)
            labels[i, :len(y)] = torch.from_numpy(y)
            mask[i, :len(x)] = 1
        return input_ids, mask, labels


class RandomChunk(Dataset):
    """One random chunk per text each time the text is drawn (a new chunk every epoch)."""

    def __init__(self, chunks: Chunks, token_lists, label_lists):
        self.c, self.x, self.y = chunks, token_lists, label_lists

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, body = self.x[i], self.c.body
        start = int(torch.randint(0, max(len(x) - body, 0) + 1, (1,)))    # torch RNG: differs per worker
        return self.c.wrap(x[start:start + body]), self.c.wrap_labels(self.y[i][start:start + body])


@torch.no_grad()
def token_probs(model, chunks: Chunks, token_lists, batch_size: int, device: str = "cuda") -> list[np.ndarray]:
    """AI probability of every token of every text (the last chunk overwrites its overlap with the one before)."""
    body = chunks.body
    parts = [(d, s, x[s:s + body]) for d, x in enumerate(token_lists) for s in chunk_starts(len(x), body)]
    order = np.argsort([len(c[2]) for c in parts], kind="stable")      # similar lengths together: less padding
    loader = DataLoader([(chunks.wrap(parts[i][2]), np.zeros(0, dtype=np.int64)) for i in order],
                        batch_size=batch_size, collate_fn=chunks.collate, num_workers=2)
    out = [np.zeros(len(x), dtype=np.float32) for x in token_lists]
    model.eval()
    pos = 0
    for input_ids, mask, _ in loader:
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        p = torch.softmax(logits.float(), dim=-1)[..., 1].cpu().numpy()
        for row in range(len(p)):
            d, s, c = parts[order[pos]]
            out[d][s:s + len(c)] = p[row, 1:1 + len(c)]
            pos += 1
    return out


def token_scores(probs, labels) -> dict:
    """Token-level precision, recall and F1 of the AI class over all texts with known labels."""
    y = np.concatenate([l for l in labels if l is not None])
    p = np.concatenate([pr for pr, l in zip(probs, labels) if l is not None]) > 0.5
    tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"token_precision": prec, "token_recall": rec, "token_f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def save_tokens(path: Path, ids, probs, starts) -> None:
    """Token probabilities (float16) and first-character positions of every text in one compressed file."""
    lengths = np.array([len(p) for p in probs])
    np.savez_compressed(path, id=np.asarray(ids, dtype=str), offsets=np.concatenate([[0], np.cumsum(lengths)]),
                        prob=np.concatenate(probs).astype(np.float16), start=np.concatenate(starts))


def load_tokens(path: Path):
    """Inverse of save_tokens: (ids, list of probability arrays, list of first-character arrays)."""
    z = np.load(path)
    o = z["offsets"]
    prob, start = z["prob"].astype(np.float32), z["start"]
    return (z["id"], [prob[a:b] for a, b in zip(o[:-1], o[1:])], [start[a:b] for a, b in zip(o[:-1], o[1:])])


def _smooth(p: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(p, np.ones(window) / window, mode="valid") if len(p) >= window else np.array([p.mean()])


def tag_features(p: np.ndarray, starts: np.ndarray | None = None, text: str | None = None,
                 windows=(32,), fine: bool = False) -> dict:
    """Per-text summaries of token AI probabilities.

    The first eight are the features of notebook 10 (smoothing window 32). With fine=True, finer ones are added:
    smoothing over shorter windows, tokens above 0.9, and sentence-level means (sentences found by end punctuation;
    needs `starts` and `text`).
    """
    hard = np.concatenate([[False], p > 0.5, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(hard))                  # starts and ends of runs of AI tokens
    runs = edges[1::2] - edges[0::2]
    longest = runs.max() if len(runs) else 0
    s32 = _smooth(p, 32)
    f = {"tag_mean": p.mean(), "tag_share": (p > 0.5).mean(), "tag_std": p.std(), "tag_runs": len(runs),
         "tag_longest_share": longest / len(p), "tag_longest_log": np.log1p(longest),
         "tag_smooth_max": s32.max(), "tag_smooth_min": s32.min()}
    if not fine:
        return f
    for w in windows:
        if w != 32:
            s = _smooth(p, w)
            f[f"tag_smooth{w}_max"], f[f"tag_smooth{w}_min"] = s.max(), s.min()
    f["tag_sure_log"] = np.log1p((p > 0.9).sum())          # number of tokens the tagger is sure about
    f["tag_top10_mean"] = np.sort(p)[-10:].mean()           # mean of the 10 most AI-looking tokens
    sent_start = np.array([0] + [m.end() for m in SENTENCE_END.finditer(text)])
    sent = np.searchsorted(sent_start, starts, side="right") - 1          # sentence index of every token
    means = np.bincount(sent, weights=p) / np.maximum(np.bincount(sent), 1)
    means = means[np.bincount(sent) > 0]
    f["sent_max"] = means.max()
    f["sent_ai_count_log"] = np.log1p((means > 0.5).sum())
    f["sent_ai_share"] = (means > 0.5).mean()
    return f
