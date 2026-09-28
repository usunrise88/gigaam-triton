#!/usr/bin/env python3
"""Rough GigaAM vs T-one comparison on the supplied audio (spec §10, reduced).

This is not the full eval module: no subsets, no substitution matrix, no
short-reply context sweep. It answers one question on a handful of files -- is
GigaAM better or worse than what is in production, and by roughly how much.

Two things it does not cut corners on, because cutting them would make the
numbers meaningless rather than rough:

* T-one is driven through its own client (spec §10.4), not a reimplementation.
* Both sides and the reference go through the same normalisation (spec §10.3),
  and WER is reported raw as well, because a large gap between the two is a sign
  the normaliser is wrong rather than that a model is good.

  PYTHONPATH=eval/_deps SecondChain/venv/bin/python eval/run_eval.py \\
      --audio-dir eval/audio --gigaam-manifest /tmp/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "_deps"))

import numpy as np  # noqa: E402

import audio_io  # noqa: E402
import normalize as norm  # noqa: E402


def collect(audio_dir: Path) -> list[tuple[Path, str]]:
    items = []
    for wav in sorted(audio_dir.glob("*.wav")):
        ref = wav.with_suffix(".txt")
        if not ref.exists():
            print(f"  skip {wav.name}: no reference .txt")
            continue
        items.append((wav, ref.read_text(encoding="utf-8").strip()))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="eval/audio")
    parser.add_argument("--gigaam-url", default="localhost:18001")
    parser.add_argument("--gigaam-manifest", default=None)
    parser.add_argument("--gigaam-label", default=None)
    parser.add_argument("--tone-url", default="localhost:17001")
    parser.add_argument("--tone-decoders", default="beam_search,greedy")
    parser.add_argument("--kenlm", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-tone", action="store_true")
    parser.add_argument("--skip-gigaam", action="store_true")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    items = collect(audio_dir)
    if not items:
        print("no audio with references found", file=sys.stderr)
        return 1
    print(f"{len(items)} file(s) with references\n")

    engines = []
    if not args.skip_gigaam and args.gigaam_manifest:
        from engines.gigaam_triton import GigaAMTritonEngine

        engines.append(
            GigaAMTritonEngine(args.gigaam_url, args.gigaam_manifest, args.gigaam_label)
        )
    if not args.skip_tone:
        from engines.tone_triton import ToneTritonEngine

        for decoder in [d.strip() for d in args.tone_decoders.split(",") if d.strip()]:
            engines.append(ToneTritonEngine(args.tone_url, decoder=decoder, kenlm_path=args.kenlm))

    if not engines:
        print("no engines selected", file=sys.stderr)
        return 1

    rows = []
    for wav, reference in items:
        audio, sr, info = audio_io.load(wav)
        duration = audio.shape[0] / sr

        audio_16k = audio_io.resample(audio, sr, audio_io.GIGAAM_SR)
        audio_8k = audio_io.resample(audio, sr, audio_io.TONE_SR)

        note = ""
        if info.get("two_real_channels"):
            note = "  [stereo with distinct channels, mixed to mono]"
        print(f"--- {wav.name}  {duration:.1f}s{note}")
        print(f"    ref : {reference}")

        spans = None
        for engine in engines:
            started = time.perf_counter()
            if engine.name.startswith("gigaam"):
                if spans is None:
                    spans = audio_io.segment(
                        audio_16k, audio_io.GIGAAM_SR,
                        max_s=engine.max_segment_samples / audio_io.GIGAAM_SR,
                    )
                    if len(spans) > 1:
                        print(f"    (segmented into {len(spans)} chunks -- GigaAM is "
                              f"offline with a 25 s ceiling, spec §11.1)")
                hypothesis = engine.transcribe(audio_16k, spans)
            else:
                hypothesis = engine.transcribe(audio_8k)
            elapsed = time.perf_counter() - started

            raw_wer = norm.wer(reference, hypothesis)
            n_ref = norm.normalize(reference)
            n_hyp = norm.normalize(hypothesis)
            rows.append(
                {
                    "file": wav.name,
                    "duration_s": round(duration, 2),
                    "engine": engine.name,
                    "hypothesis": hypothesis,
                    "normalised": n_hyp,
                    "wer_raw": round(raw_wer, 4),
                    "wer": round(norm.wer(n_ref, n_hyp), 4),
                    "cer": round(norm.cer(n_ref, n_hyp), 4),
                    "wer_no_hesitations": round(
                        norm.wer(
                            norm.normalize(reference, drop_hesitations=True),
                            norm.normalize(hypothesis, drop_hesitations=True),
                        ),
                        4,
                    ),
                    "seconds": round(elapsed, 2),
                    "rtfx": round(duration / elapsed, 1) if elapsed else None,
                }
            )
            print(f"    {engine.name:<22} WER {rows[-1]['wer']:.3f} "
                  f"(raw {rows[-1]['wer_raw']:.3f})  CER {rows[-1]['cer']:.3f}  "
                  f"RTFx {rows[-1]['rtfx']}")
            print(f"      {hypothesis}")
        print()

    print("=" * 78)
    print(f"{'engine':<24}{'WER':>8}{'CER':>8}{'WER raw':>10}{'WER no-hes':>12}{'RTFx':>8}")
    print("-" * 78)
    summary = []
    for engine in engines:
        mine = [r for r in rows if r["engine"] == engine.name]
        if not mine:
            continue
        # Aggregate weighted by reference length, not a mean of per-file rates:
        # a 3-word file must not weigh the same as a 300-word one.
        refs = [norm.normalize(dict(items)[audio_dir / r["file"]]) for r in mine]
        total_words = sum(len(r.split()) for r in refs)
        wer_all = sum(r["wer"] * len(ref.split()) for r, ref in zip(mine, refs)) / max(1, total_words)
        cer_all = sum(r["cer"] * len(ref) for r, ref in zip(mine, refs)) / max(1, sum(len(r) for r in refs))
        raw_all = sum(r["wer_raw"] * len(ref.split()) for r, ref in zip(mine, refs)) / max(1, total_words)
        noh_all = sum(r["wer_no_hesitations"] * len(ref.split()) for r, ref in zip(mine, refs)) / max(1, total_words)
        rtfx = float(np.median([r["rtfx"] for r in mine if r["rtfx"]]))
        summary.append({"engine": engine.name, "wer": round(wer_all, 4), "cer": round(cer_all, 4),
                        "wer_raw": round(raw_all, 4), "wer_no_hesitations": round(noh_all, 4),
                        "rtfx_median": rtfx})
        print(f"{engine.name:<24}{wer_all:>8.3f}{cer_all:>8.3f}{raw_all:>10.3f}"
              f"{noh_all:>12.3f}{rtfx:>8.1f}")

    print("\nWER raw is scored without normalisation. A large gap between it and WER")
    print("is expected here -- the references are lowercase without punctuation while")
    print("GigaAM's e2e head emits cased, punctuated text with digits -- but if the")
    print("gap ever looks small for GigaAM, the normaliser has stopped working.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
