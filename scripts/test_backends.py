#!/usr/bin/env python3
"""Unit checks for the pieces of the Python backends that are pure logic.

The Triton backends themselves need a running server; the collapse, the
tokenizer and the input validation do not, and those are where a quiet mistake
would cost the most -- a wrong collapse produces plausible text, not a crash.

  python scripts/test_backends.py --onnx-root artifacts/onnx --variant v3_e2e_ctc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.common import ctc  # noqa: E402
from backends.common.logmel import LogMel  # noqa: E402
from backends.common.tokenizer import Tokenizer  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(f"{name} {detail}".strip())


def test_collapse() -> None:
    print("\nCTC collapse:")
    blank = 256

    ids = np.array([1, 1, blank, 1, 2, 2, blank, 3], dtype=np.int32)
    scores = np.arange(8, dtype=np.float32)
    tokens, sc = ctc.collapse(ids, scores, len(ids), blank)
    check("repeats collapsed, blanks dropped", tokens == [1, 1, 2, 3], f"got {tokens}")
    check("scores taken at emission frames", sc == [0.0, 3.0, 4.0, 7.0], f"got {sc}")

    # A repeat separated by a blank is two tokens; without the blank it is one.
    # Getting this backwards silently doubles or halves letters.
    tokens, _ = ctc.collapse(np.array([5, 5], np.int32), np.zeros(2, np.float32), 2, blank)
    check("adjacent repeat is one token", tokens == [5], f"got {tokens}")
    tokens, _ = ctc.collapse(
        np.array([5, blank, 5], np.int32), np.zeros(3, np.float32), 3, blank
    )
    check("repeat across a blank is two", tokens == [5, 5], f"got {tokens}")

    # length shorter than the tensor: everything past it is padding from the
    # bucket and must not be decoded.
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    tokens, _ = ctc.collapse(ids, np.zeros(5, np.float32), 2, blank)
    check("honours encoded_lengths", tokens == [1, 2], f"got {tokens}")

    tokens, _ = ctc.collapse(ids, np.zeros(5, np.float32), 0, blank)
    check("zero length yields nothing", tokens == [], f"got {tokens}")

    tokens, _ = ctc.collapse(
        np.full(4, blank, np.int32), np.zeros(4, np.float32), 4, blank
    )
    check("all blank yields nothing", tokens == [], f"got {tokens}")

    # A length longer than the tensor must clamp rather than read past the end.
    tokens, _ = ctc.collapse(ids, np.zeros(5, np.float32), 99, blank)
    check("over-long length clamps", tokens == [1, 2, 3, 4, 5], f"got {tokens}")


def test_tokenizer(onnx_dir: Path) -> None:
    print("\ntokenizer:")
    tok = Tokenizer.from_dir(str(onnx_dir))
    check("loaded from export artefacts", len(tok) > 0, f"vocab {len(tok)}")
    check("blank sits past the vocabulary", tok.blank_id == len(tok))

    meta = __import__("json").loads((onnx_dir / "meta.json").read_text())
    check(
        "vocab agrees with meta.json",
        len(tok) == meta["vocab_size"],
        f"{len(tok)} vs {meta['vocab_size']}",
    )
    check(
        "num_classes is vocab + 1",
        meta["num_classes"] == meta["vocab_size"] + 1,
        f"{meta['num_classes']}",
    )
    check("decodes an empty sequence", tok.decode([]) == "")


def test_logmel(fb_path: Path) -> None:
    print("\nlog-mel replay:")
    lm = LogMel.from_npz(str(fb_path))

    # out_len must match gigaam.preprocess.FeatureExtractor for the geometry the
    # model actually uses -- this is what every bucket shape is derived from.
    for seconds, expected in [(2.0, 199), (5.0, 499), (10.0, 999), (20.0, 1999)]:
        got = int(lm.out_len(np.array([int(seconds * 16000)]))[0])
        check(f"out_len({seconds:g}s)", got == expected, f"{got} vs {expected}")

    audio = (np.random.default_rng(0).standard_normal(32000) * 0.1).astype(np.float32)
    feats = lm(audio, dtype=np.float16)
    check("shape is (n_mels, frames)", feats.shape == (lm.n_mels, 199), str(feats.shape))
    check("emits the requested dtype", feats.dtype == np.float16, str(feats.dtype))
    check("finite on ordinary audio", bool(np.all(np.isfinite(feats))))

    silence = lm(np.zeros(16000, dtype=np.float32))
    check(
        "silence clamps instead of -inf",
        bool(np.all(np.isfinite(silence))),
        f"min {float(silence.min()):.2f}",
    )

    batched = lm(np.stack([audio, audio]))
    check("batched matches single", np.allclose(batched[0], batched[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-root", default="artifacts/onnx")
    parser.add_argument("--variant", default="v3_e2e_ctc")
    args = parser.parse_args()

    root = Path(args.onnx_root)
    onnx_dir = root / args.variant

    test_collapse()
    if onnx_dir.exists():
        test_tokenizer(onnx_dir)
        meta = __import__("json").loads((onnx_dir / "meta.json").read_text())
        fb = root / "preprocessing" / meta["family"] / "filterbank.npz"
        if fb.exists():
            test_logmel(fb)
        else:
            print(f"\n(no {fb} -- skipping log-mel checks)")
    else:
        print(f"\n(no export at {onnx_dir} -- skipping tokenizer and log-mel checks)")

    print()
    if failures:
        print("FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("all backend unit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
