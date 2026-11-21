"""Which words of a fusion text were written by the AI: a word-level diff against its human source text.

A fusion text is its human source with some sentences/paragraphs replaced by AI-written text (GhostWriter,
Section 3.2). Words that belong to a run of at least `MIN_KEPT` consecutive words shared with the source are
human; all other words of the fusion text are AI.

    spans = ai_spans(source_text, fusion_text)       # [(start_char, end_char), ...] in the fusion text
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

WORD = re.compile(r"\S+")
MIN_KEPT = 3            # shorter shared runs ("of the", "and") inside a rewritten section do not count as kept


def ai_word_mask(source: str, fusion: str) -> list[bool]:
    """True for every word of `fusion` that is not part of a long enough run shared with `source`."""
    src, fus = source.split(), fusion.split()
    ai = [True] * len(fus)
    for block in SequenceMatcher(None, src, fus, autojunk=False).get_matching_blocks():
        if block.size >= MIN_KEPT:
            ai[block.b:block.b + block.size] = [False] * block.size
    return ai


def ai_spans(source: str, fusion: str) -> list[tuple[int, int]]:
    """Character spans of the AI-written runs of words in `fusion`."""
    words = list(WORD.finditer(fusion))
    spans, start = [], None
    for m, is_ai in zip(words, ai_word_mask(source, fusion)):
        if is_ai and start is None:
            start = m.start()
        if not is_ai and start is not None:
            spans.append((start, prev_end))
            start = None
        prev_end = m.end()
    if start is not None:
        spans.append((start, prev_end))
    return spans


def best_source(fusion: str, candidates: list[str]) -> int:
    """Index of the candidate human text that shares the most words with `fusion` (cheap bag-of-words ratio)."""
    fus = fusion.split()
    return max(range(len(candidates)),
               key=lambda i: SequenceMatcher(None, candidates[i].split(), fus, autojunk=False).quick_ratio())


def span_record(args: tuple[str, str]) -> dict:
    """Worker for multiprocessing: spans and summary numbers for one (source, fusion) pair."""
    source, fusion = args
    spans = ai_spans(source, fusion)
    n_words = len(fusion.split())
    ai_words = sum(len(fusion[s:e].split()) for s, e in spans)
    return {"n_words": n_words, "ai_share": ai_words / max(n_words, 1), "n_spans": len(spans),
            "first_ai": spans[0][0] / max(len(fusion), 1) if spans else float("nan"),
            "spans": ";".join(f"{s}:{e}" for s, e in spans)}


def parse_spans(s) -> list[tuple[int, int]]:
    """Inverse of the `spans` string of span_record."""
    if not isinstance(s, str) or not s:
        return []
    return [tuple(map(int, p.split(":"))) for p in s.split(";")]
