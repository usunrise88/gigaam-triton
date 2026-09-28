#!/usr/bin/env python3
"""Gate: the torch-free numpy log-mel must match torchaudio's.

The runtime image has no torch, so the Triton preprocessing backend recomputes
log-mel with numpy over the filterbank the builder exported. If that replay
drifts, nothing crashes -- the model just gets slightly wrong features and the
WER regression gets blamed on the model. So this runs in the builder, before
anything else is produced, and fails the build on mismatch.

Compared against ``gigaam.preprocess.FeatureExtractor`` itself, not against a
second implementation of the same idea.

  python scripts/check_logmel.py --variant v3_e2e_ctc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fbank_export  # noqa: E402
from backends.common.logmel import LogMel  # noqa: E402

SAMPLE_RATE = 16000

# Compared against torchaudio in float64, not float32.
#
# The first version of this gate compared against the float32 path and failed a
# chirp at 3.8e-3 -- which turned out to be torch's own float32 rounding on a mel
# bin holding 1.8e-8 of energy, where the log amplifies relative error. Measured
# against a float64 reference the replay agrees to 2.4e-11, i.e. it is exact and
# the float32 path is the noisier of the two.
#
# So the tolerance is set at the level of a correct implementation, not at the
# level of the noise it is compared with. Anything above this is a real
# divergence in the filterbank, window, padding or framing.
MAX_ABS_DIFF = 1e-8


def synthetic_cases() -> list[tuple[str, np.ndarray]]:
    """Signals chosen to stress the parts most likely to diverge: the reflect
    padding at the edges, the log clamp on silence, and full-scale content."""
    rng = np.random.default_rng(1234)
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    return [
        ("silence_1s", np.zeros(SAMPLE_RATE, dtype=np.float32)),
        ("white_noise_3s", rng.standard_normal(SAMPLE_RATE * 3).astype(np.float32) * 0.1),
        ("sine_440_3s", (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)),
        ("chirp_3s", np.sin(2 * np.pi * (50 + 3000 * t) * t).astype(np.float32) * 0.5),
        ("full_scale_1s", np.ones(SAMPLE_RATE, dtype=np.float32)),
        # Short clip: the `short` eval subset is where candidates historically
        # broke, and it is also where reflect padding dominates the frame count.
        ("very_short_300ms", rng.standard_normal(4800).astype(np.float32) * 0.2),
        ("tiny_50ms", rng.standard_normal(800).astype(np.float32) * 0.2),
    ]


def golden_cases(golden_dir: Path) -> list[tuple[str, np.ndarray]]:
    cases: list[tuple[str, np.ndarray]] = []
    if not golden_dir.is_dir():
        return cases
    import soundfile as sf

    for wav in sorted(golden_dir.glob("*.wav")):
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            print(f"  skip {wav.name}: {sr} Hz, expected {SAMPLE_RATE}")
            continue
        cases.append((wav.name, audio.astype(np.float32)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="v3_e2e_ctc")
    parser.add_argument("--golden-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=MAX_ABS_DIFF)
    parser.add_argument("--filterbank-out", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    golden_dir = Path(args.golden_dir or repo_root / "testdata" / "golden")

    import gigaam
    from gigaam.preprocess import FeatureExtractor

    print(f"loading {args.variant} (cpu) ...")
    model = gigaam.load_model(args.variant, device="cpu", fp16_encoder=False)
    reference: FeatureExtractor = model.preprocessor
    reference.eval()

    fb_path = args.filterbank_out or str(repo_root / "cache" / "filterbank.npz")
    Path(fb_path).parent.mkdir(parents=True, exist_ok=True)
    fbank_export.save(reference, fb_path)
    print(f"filterbank exported -> {fb_path}")

    replay = LogMel.from_npz(fb_path)
    print(
        f"geometry: n_fft={replay.n_fft} hop={replay.hop_length} "
        f"win={replay.window.shape[0]} center={replay.center} n_mels={replay.n_mels}"
    )

    # float64 reference. The production path runs float32, but its rounding is
    # larger than the quantity we are trying to measure, so it cannot serve as
    # the yardstick for our own correctness.
    reference_f64 = reference.double()

    cases = synthetic_cases() + golden_cases(golden_dir)
    print(f"comparing {len(cases)} cases, tolerance {args.tolerance:g}\n")

    worst = 0.0
    worst_f32_noise = 0.0
    failures: list[str] = []

    for name, audio in cases:
        lengths = torch.tensor([len(audio)], dtype=torch.long)
        with torch.no_grad():
            ref_feats, ref_len = reference_f64(
                torch.from_numpy(audio)[None, :].double(), lengths
            )
            f32_feats, _ = reference_f64.float()(
                torch.from_numpy(audio)[None, :].float(), lengths
            )
            reference_f64.double()
        ref = ref_feats.numpy()[0]

        got = replay(audio, dtype=np.float64)
        got_len = replay.out_len(np.array([len(audio)]))[0]

        if got.shape != ref.shape:
            failures.append(f"{name}: shape {got.shape} != torch {ref.shape}")
            continue
        if int(got_len) != int(ref_len[0]):
            failures.append(f"{name}: out_len {int(got_len)} != torch {int(ref_len[0])}")
            continue

        diff = float(np.max(np.abs(got - ref)))
        f32_noise = float(np.max(np.abs(f32_feats.numpy()[0].astype(np.float64) - ref)))
        worst = max(worst, diff)
        worst_f32_noise = max(worst_f32_noise, f32_noise)

        status = "ok " if diff <= args.tolerance else "FAIL"
        if diff > args.tolerance:
            failures.append(f"{name}: max abs diff {diff:.3e} > {args.tolerance:g}")
        print(
            f"  {status} {name:<20} frames={ref.shape[1]:>5}  "
            f"vs_f64={diff:.3e}  (torch f32 noise {f32_noise:.1e})"
        )

    print()
    if failures:
        print("numpy log-mel does NOT match torchaudio:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nRefusing to build a runtime whose front end differs from the model's.\n"
            "Either fix backends/common/logmel.py or keep torch in the runtime image\n"
            "(an explicit deviation from spec §4.2, not a silent workaround).",
            file=sys.stderr,
        )
        return 1

    print(
        f"PASSED -- worst diff vs torch-float64 {worst:.3e} "
        f"(tolerance {args.tolerance:g}); torch's own float32 path is off by up to "
        f"{worst_f32_noise:.1e}, so the replay is the more accurate of the two."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
