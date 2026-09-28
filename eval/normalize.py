"""Text normalisation for the comparison (spec §10.3).

This is the step that decides whether the whole comparison means anything. T-one
emits lowercase with no punctuation; GigaAM's e2e heads emit cased, punctuated
text with numbers written as digits. Scoring those against each other raw counts
"двадцать пять" against "25" as two errors that neither model made.

Both sides and the reference go through the same pipeline, and numbers collapse
towards words rather than digits: going the other way (inverse text
normalisation) is ambiguous -- "двадцать пять" could be 25 or 20 5 -- while
spelling a digit out is deterministic.
"""

from __future__ import annotations

import re
import unicodedata

from num2words import num2words

# Hesitations. Removing them is optional and the report shows both, because
# whether they count depends on what the transcript is for.
HESITATIONS = {"э", "ээ", "эээ", "м", "мм", "ммм", "а-а", "э-э", "гм", "кхм"}

_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
_DASHES = re.compile(r"[‐-―−]")
_SPACES = re.compile(r"\s+")


def _number_to_words(match: re.Match) -> str:
    raw = match.group(0)
    try:
        if "," in raw or "." in raw:
            whole, frac = re.split(r"[.,]", raw, maxsplit=1)
            # "3,5" -> "три целых пять десятых" is over-engineering for a rough
            # comparison; read it as two numbers, which is how it is usually said.
            return f"{num2words(int(whole), lang='ru')} {num2words(int(frac), lang='ru')}"
        return num2words(int(raw), lang="ru")
    except (ValueError, OverflowError):
        return raw


def normalize(text: str, drop_hesitations: bool = False) -> str:
    """Bring any of the three text sources to one comparable form."""
    if not text:
        return ""

    text = text.replace("ё", "е").replace("Ё", "Е")
    text = _DASHES.sub(" ", text)
    text = text.lower()

    # Digits before punctuation stripping, so "25%" and "1,5" are still intact.
    text = _DIGITS.sub(_number_to_words, text)

    kept = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            kept.append(ch)
        elif unicodedata.category(ch).startswith("P"):
            # Punctuation becomes a break, not a deletion: "да,нет" must not
            # become one token.
            kept.append(" ")
        else:
            kept.append(" ")
    text = "".join(kept)

    words = _SPACES.sub(" ", text).strip().split()
    if drop_hesitations:
        words = [w for w in words if w not in HESITATIONS]

    return " ".join(words)


# ---------------------------------------------------------------- metrics


def _levenshtein(a: list, b: list) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    if not ref:
        return 0.0 if not hypothesis.split() else 1.0
    return _levenshtein(ref, hypothesis.split()) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    if not ref:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(ref, list(hypothesis)) / len(ref)


def sentence_accuracy(references: list[str], hypotheses: list[str]) -> float:
    """Exact-match rate. More informative than WER on very short utterances,
    which is where candidates have historically broken (spec §10.6)."""
    if not references:
        return 0.0
    return sum(r == h for r, h in zip(references, hypotheses)) / len(references)


SELF_TESTS = [
    ("Двадцать пять рублей.", "двадцать пять рублей"),
    ("25 рублей", "двадцать пять рублей"),
    ("Да, конечно — записывайте!", "да конечно записывайте"),
    ("ёлка", "елка"),
    ("да,нет", "да нет"),
    ("В 2024 году", "в две тысячи двадцать четыре году"),
    ("?угу? ну говорю же", "угу ну говорю же"),
    ("  много   пробелов  ", "много пробелов"),
]


if __name__ == "__main__":
    import sys

    failures = 0
    for raw, expected in SELF_TESTS:
        got = normalize(raw)
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {raw!r} -> {got!r}"
              + ("" if ok else f"   expected {expected!r}"))
    # The two heads must land on the same normalised string for the same words.
    ctc = normalize("Счастлив уж я надеждой сладкой, Что дева с трепетом любви")
    rnnt = normalize("Счастлив уж я надеждой сладкой, что дева с трепетом любви.")
    print(f"  {'ok  ' if ctc == rnnt else 'FAIL'} casing/punctuation differences collapse")
    failures += ctc != rnnt
    sys.exit(1 if failures else 0)
