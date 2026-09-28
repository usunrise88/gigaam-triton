#!/usr/bin/env python3
"""End-to-end check against a running server (spec §12 steps 3-4).

Covers three things a green build does not:

1. The ensemble actually transcribes golden audio, per bucket, over gRPC.
2. Bad input fails loudly. GigaAM fed int16-scaled samples returns an empty
   string with no error (spec §11.2) -- a smoke test that only asserts "we got a
   response" would pass while the server quietly produced nothing.
3. The extra outputs from spec §7.2 (logprobs, tokens, scores) are present,
   correctly shaped, and agree with the text the server itself decoded.

  python scripts/smoke_test.py --url localhost:18001 --repo /models
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_SMOKE_CER = 0.005


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(1, len(ref))


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Polyphase, not linear interpolation, and once per segment -- resampling
        # per frame leaves filter artefacts on every boundary (spec §11.5).
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def pick_ensemble(manifest: dict, n_samples: int) -> dict | None:
    """Smallest bucket that fits -- the same choice the provider makes."""
    for entry in sorted(manifest["ensembles"], key=lambda e: e["pad_to_samples"]):
        if n_samples <= entry["pad_to_samples"]:
            return entry
    return None


def infer(client, model: str, audio: np.ndarray, pad_to: int, outputs: list[str]):
    import tritonclient.grpc as grpcclient

    padded = np.zeros(pad_to, dtype=np.float32)
    padded[: audio.shape[0]] = audio

    audio_in = grpcclient.InferInput("audio", [1, pad_to], "FP32")
    audio_in.set_data_from_numpy(padded[None, :])
    len_in = grpcclient.InferInput("audio_len", [1, 1], "INT32")
    len_in.set_data_from_numpy(np.array([[audio.shape[0]]], dtype=np.int32))

    return client.infer(
        model,
        [audio_in, len_in],
        outputs=[grpcclient.InferRequestedOutput(o) for o in outputs],
    )


def _check_logprobs(manifest, wav, logprobs, n_frames, tokens, failures) -> None:
    """CTC only: the provider decodes these itself (spec §7.2), so they must be a
    real log-softmax and must collapse to the tokens the server returned. If the
    two drift, the provider's confidence and word timings describe a transcript
    nobody actually produced."""
    if logprobs.shape[-1] != manifest["num_classes"]:
        failures.append(
            f"{wav.name}: logprobs width {logprobs.shape[-1]} != "
            f"{manifest['num_classes']} (vocab {manifest['vocab_size']} + blank)"
        )
    if n_frames <= 0 or n_frames > logprobs.shape[0]:
        failures.append(f"{wav.name}: logprobs_len {n_frames} outside {logprobs.shape}")
        return

    row_sum = float(np.exp(logprobs[0]).sum())
    if not 0.98 <= row_sum <= 1.02:
        failures.append(
            f"{wav.name}: logprobs row sums to {row_sum:.3f}, not a log-softmax"
        )

    argmax = logprobs[:n_frames].argmax(axis=-1)
    keep = argmax != manifest["blank_id"]          # blank sits past the vocab
    keep[1:] &= argmax[1:] != argmax[:-1]
    client_tokens = argmax[keep].astype(np.int32)
    if not np.array_equal(client_tokens, tokens):
        failures.append(
            f"{wav.name}: client-side collapse of logprobs gives "
            f"{len(client_tokens)} tokens, server returned {len(tokens)}"
        )


CLIP_NAME = re.compile(r"^(?P<stem>.+)_(?P<secs>\d+(?:\.\d+)?)s$")


def derive_clips(golden_dir: Path) -> list[Path]:
    """Cut the short golden clips that have a committed reference but no audio.

    ``<stem>_<N>s`` is the first N seconds of ``<stem>.wav`` (e.g. example_3s ->
    "Ничьих не требуя похвал, счастлив уж я на"). Only the references are in git --
    audio is ignored -- so a fresh build has the 11 s example.wav and nothing that
    fits a small-bucket deployment, and the golden check would skip everything.
    Clips go next to the references when the directory is writable, otherwise to a
    temporary directory; the references are always read from ``golden_dir``.
    """
    import soundfile as sf

    out_dir = golden_dir if os.access(golden_dir, os.W_OK) else None
    made: list[Path] = []
    names = {p.name.split(".", 1)[0] for p in golden_dir.glob("*.txt")}
    for name in sorted(names):
        m = CLIP_NAME.match(name)
        src = golden_dir / f"{m['stem']}.wav" if m else None
        if not m or (golden_dir / f"{name}.wav").exists() or not src.exists():
            continue
        clip = read_audio(src)[: int(float(m["secs"]) * SAMPLE_RATE)]
        if out_dir is None:  # read-only mount (e.g. -v ...:/testdata:ro)
            out_dir = Path(tempfile.mkdtemp(prefix="golden-"))
        dest = out_dir / f"{name}.wav"
        sf.write(str(dest), clip, SAMPLE_RATE, subtype="FLOAT")
        print(f"  derived {dest.name} from {src.name} (first {m['secs']} s)")
        made.append(dest)
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:18001")
    parser.add_argument("--repo", default="/models")
    parser.add_argument("--golden-dir", default=None)
    parser.add_argument("--tolerance", type=float, default=MAX_SMOKE_CER)
    args = parser.parse_args()

    import tritonclient.grpc as grpcclient

    repo_root = Path(__file__).resolve().parent.parent
    golden_dir = Path(args.golden_dir or repo_root / "testdata" / "golden")
    manifest = json.loads((Path(args.repo) / "manifest.json").read_text())
    # RNNT does not expose logprobs: a transducer lattice is not usable by the
    # provider, so confidence comes from token_scores instead (spec §7.2).
    is_rnnt = manifest.get("graph") == "rnnt"
    request_outputs = ["text", "tokens", "token_scores", "tokens_len"]
    if not is_rnnt:
        request_outputs += ["logprobs", "logprobs_len"]

    client = grpcclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        print(f"server at {args.url} is not ready", file=sys.stderr)
        return 1
    print(f"server ready: {args.url}  variant={manifest['variant']} "
          f"runtime={manifest['runtime']}\n")

    failures: list[str] = []

    # ---- 0/3: warm every ensemble --------------------------------------
    # model_warmup covers the encoder, but Triton has no warmup for ensembles,
    # and each one carries a one-time first-request cost of its own (measured at
    # 16-20 ms per ensemble, and ~70 ms for the first one touched in a process).
    # Since this script is the post-deploy gate, warming here means the first
    # production call does not pay it (spec §7.3 wants warmup to actually take).
    print("warming ensembles (Triton cannot warm these itself):")
    for entry in sorted(manifest["ensembles"], key=lambda e: e["pad_to_samples"]):
        blank = np.zeros(entry["pad_to_samples"], dtype=np.float32)
        started = time.perf_counter()
        try:
            infer(client, entry["name"], blank, entry["pad_to_samples"], ["text"])
            print(f"  {entry['name']:<32} {(time.perf_counter() - started) * 1000:7.1f} ms")
        except Exception as exc:
            failures.append(f"{entry['name']}: warm-up request failed: {exc}")
            print(f"  {entry['name']:<32} FAILED")
    print()

    # ---- 1/3: golden transcription -------------------------------------
    wavs = sorted(set(golden_dir.glob("*.wav")) | set(derive_clips(golden_dir)), key=lambda p: p.name)
    if not wavs:
        failures.append(f"no golden audio in {golden_dir}")
    checked = 0
    for wav in wavs:
        audio = read_audio(wav)
        entry = pick_ensemble(manifest, audio.shape[0])
        if entry is None:
            # Not a defect: a deployment with small buckets legitimately cannot
            # take a long clip in one piece, and the client is expected to
            # segment. Skip it loudly rather than failing on a file that is only
            # too long for this configuration -- but see the check below, which
            # fails if that leaves nothing tested at all.
            print(f"  skip {wav.name}: {audio.shape[0] / SAMPLE_RATE:.1f}s exceeds the "
                  f"largest bucket ({manifest['ensembles'][-1]['bucket_s']:g}s)")
            continue
        checked += 1

        result = infer(
            client, entry["name"], audio, entry["pad_to_samples"], request_outputs
        )
        text = result.as_numpy("text")[0][0].decode("utf-8")

        n_tokens = int(result.as_numpy("tokens_len")[0][0])
        tokens = result.as_numpy("tokens")[0][:n_tokens]
        scores = result.as_numpy("token_scores")[0][:n_tokens]

        print(f"  {wav.name} -> {entry['name']}")
        print(f"    text: {text!r}")

        if not text.strip():
            failures.append(f"{wav.name}: empty transcript")

        if is_rnnt:
            print(f"    tokens={n_tokens} "
                  f"mean_score={float(np.mean(scores)) if n_tokens else float('nan'):.3f}")
            if n_tokens <= 0:
                failures.append(f"{wav.name}: transducer emitted no tokens")
        else:
            logprobs = result.as_numpy("logprobs")[0]
            n_frames = int(result.as_numpy("logprobs_len")[0][0])
            print(f"    frames={n_frames} tokens={n_tokens} vocab={logprobs.shape[-1]} "
                  f"mean_score={float(np.mean(scores)) if n_tokens else float('nan'):.3f}")
            _check_logprobs(manifest, wav, logprobs, n_frames, tokens, failures)

        # References are per variant. The heads genuinely disagree -- CTC and
        # RNNT punctuate the same audio differently -- so checking one against
        # the other's reference measures nothing and fails for the wrong reason.
        expected_path = golden_dir / f"{wav.stem}.{manifest['variant']}.txt"
        if not expected_path.exists():
            expected_path = golden_dir / f"{wav.stem}.txt"
        if expected_path.exists():
            expected = expected_path.read_text(encoding="utf-8").strip()
            score = cer(expected, text.strip())
            mark = "ok " if score <= args.tolerance else "FAIL"
            print(f"    {mark} CER vs reference: {score:.4f}")
            if score > args.tolerance:
                failures.append(f"{wav.name}: CER {score:.4f} > {args.tolerance}")
        else:
            print(f"    (no {expected_path.name} -- transcript not checked)")

    if wavs and not checked:
        failures.append(
            f"every golden file is longer than the largest bucket "
            f"({max(e['bucket_s'] for e in manifest['ensembles']):g}s) -- nothing was "
            "actually transcribed. Add a clip that fits this deployment."
        )

    # ---- 2/3: bad input must fail loudly -------------------------------
    print("\nnegative tests (spec §11.2):")
    entry = sorted(manifest["ensembles"], key=lambda e: e["pad_to_samples"])[0]
    rng = np.random.default_rng(7)
    base = (rng.standard_normal(entry["pad_to_samples"] // 2) * 0.05).astype(np.float32)

    negatives = [
        ("int16-scaled samples", base * 32768.0, base.shape[0]),
        ("audio_len beyond buffer", base, entry["pad_to_samples"] * 4),
        ("NaN in audio", np.where(np.arange(base.shape[0]) == 5, np.nan, base).astype(np.float32),
         base.shape[0]),
    ]

    for label, bad_audio, bad_len in negatives:
        padded = np.zeros(entry["pad_to_samples"], dtype=np.float32)
        padded[: bad_audio.shape[0]] = bad_audio
        try:
            import tritonclient.grpc as gc

            a = gc.InferInput("audio", [1, entry["pad_to_samples"]], "FP32")
            a.set_data_from_numpy(padded[None, :])
            n = gc.InferInput("audio_len", [1, 1], "INT32")
            n.set_data_from_numpy(np.array([[bad_len]], dtype=np.int32))
            out = client.infer(entry["name"], [a, n],
                               outputs=[gc.InferRequestedOutput("text")])
            text = out.as_numpy("text")[0][0].decode("utf-8")
            failures.append(
                f"{label}: server returned {text!r} instead of an error -- this is "
                "exactly the silent failure spec §11.2 is about"
            )
            print(f"  FAIL {label}: got {text!r}")
        except Exception as exc:
            print(f"  ok   {label}: rejected ({str(exc).splitlines()[0][:90]})")

    # ---- 3/3: every bucket loads ---------------------------------------
    print("\nbucket availability:")
    for e in manifest["ensembles"]:
        ready = client.is_model_ready(e["name"])
        print(f"  {'ok  ' if ready else 'FAIL'} {e['name']}")
        if not ready:
            failures.append(f"{e['name']} not ready")

    print()
    if failures:
        print("SMOKE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
