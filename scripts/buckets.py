"""Duration buckets shared by the engine builder and the repo generator.

These two must agree exactly. ``build_trt.py`` emits one TensorRT optimization
profile per bucket, in order, and ``gen_model_repo.py`` pins each Triton model to
a profile by index. If they disagree, requests land on a profile whose shape
range does not cover them and the failure reads as a runtime shape error rather
than a configuration mistake. So the table lives in one file.

Frame counts are derived from the model's own feature geometry, never assumed.
v3 does not use the library defaults -- it runs n_fft=320 with center=False,
where v2 uses 400 with centring -- so the centred formula would be off by two
frames per bucket and requests at exactly the bucket maximum would fall outside
their profile. This is the same v2/v3 divergence spec §5.2 and §11.3 are about,
arriving through the shape math instead of through the filterbank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

SAMPLE_RATE = 16000
MIN_FIRST_BUCKET_S = 0.3  # shortest clip we are willing to build a profile for

DEFAULT_BUCKETS_S = [2, 5, 10, 20]

# The default set is the table from spec §6.1 verbatim, tuned for phone dialogue.
# Its opt points are not a formula -- they sit where the traffic sits -- so they
# are written out rather than derived.
_SPEC_OPT_S = {2: 1.5, 5: 3.5, 10: 7.0, 20: 14.0}


@dataclass(frozen=True)
class FrameGeometry:
    """Mirrors gigaam.preprocess.FeatureExtractor.out_len."""

    sample_rate: int = SAMPLE_RATE
    hop_length: int = 160
    win_length: int = 320
    center: bool = False

    def frames(self, seconds: float) -> int:
        n = int(round(seconds * self.sample_rate))
        if self.center:
            return n // self.hop_length + 1
        return (n - self.win_length) // self.hop_length + 1

    @classmethod
    def from_meta(cls, meta: dict) -> "FrameGeometry":
        missing = [k for k in ("hop_length", "win_length", "center") if k not in meta]
        if missing:
            raise SystemExit(
                f"meta.json is missing feature geometry {missing}. It is written by "
                "scripts/export_onnx.py -- re-export rather than guessing, because a "
                "wrong frame count silently shifts every optimization profile."
            )
        return cls(
            sample_rate=int(meta.get("sample_rate", SAMPLE_RATE)),
            hop_length=int(meta["hop_length"]),
            win_length=int(meta["win_length"]),
            center=bool(meta["center"]),
        )


@dataclass(frozen=True)
class Bucket:
    index: int
    max_s: float
    min_s: float
    opt_s: float
    geometry: FrameGeometry

    @property
    def label(self) -> str:
        return f"{self.max_s:g}"

    @property
    def frames_min(self) -> int:
        return self.geometry.frames(self.min_s)

    @property
    def frames_opt(self) -> int:
        return self.geometry.frames(self.opt_s)

    @property
    def frames_max(self) -> int:
        return self.geometry.frames(self.max_s)

    @property
    def samples_max(self) -> int:
        return int(self.max_s * self.geometry.sample_rate)


def parse(spec: str | None) -> List[float] | None:
    """``"2,5,10,20"`` -> [2, 5, 10, 20]; ``"none"`` -> None (single wide profile)."""
    if spec is None:
        return DEFAULT_BUCKETS_S
    text = spec.strip().lower()
    if text in {"none", "off", ""}:
        return None
    values = [float(p) for p in text.split(",") if p.strip()]
    if not values:
        raise ValueError(f"could not parse buckets from {spec!r}")
    if sorted(values) != values:
        raise ValueError(f"buckets must be ascending, got {values}")
    return values


def build(
    bucket_s: List[float] | None,
    max_audio_s: float = 25.0,
    geometry: FrameGeometry | None = None,
) -> List[Bucket]:
    """Bucket list with min/opt/max ranges.

    ``None`` yields the single wide profile -- the baseline bucketing has to beat
    on measurement before the extra complexity is worth it (spec §6.1 asks for
    the justification, §1 asks for a measurement rather than an argument).
    """
    geom = geometry or FrameGeometry()

    if bucket_s is None:
        return [
            Bucket(
                index=0,
                min_s=MIN_FIRST_BUCKET_S,
                opt_s=round(max_audio_s / 2, 3),
                max_s=max_audio_s,
                geometry=geom,
            )
        ]

    buckets: List[Bucket] = []
    lower = MIN_FIRST_BUCKET_S
    for i, upper in enumerate(bucket_s):
        if upper <= lower:
            raise ValueError(f"bucket {upper} not greater than previous bound {lower}")
        # Custom sets get the midpoint; only the default table has hand-picked
        # opt points, and inventing a formula to reproduce them would be fiction.
        opt = _SPEC_OPT_S.get(upper, round((lower + upper) / 2, 3))
        buckets.append(
            Bucket(index=i, min_s=lower, opt_s=opt, max_s=float(upper), geometry=geom)
        )
        lower = float(upper)

    return buckets


def describe(buckets: List[Bucket]) -> str:
    g = buckets[0].geometry
    head = (
        f"feature geometry: hop={g.hop_length} win={g.win_length} "
        f"center={g.center} sr={g.sample_rate}"
    )
    rows = [head, f"{'bucket':>8} {'min':>7} {'opt':>7} {'max':>7}   frames min/opt/max"]
    for b in buckets:
        rows.append(
            f"{b.label + 's':>8} {b.min_s:>7.2f} {b.opt_s:>7.2f} {b.max_s:>7.2f}   "
            f"{b.frames_min} / {b.frames_opt} / {b.frames_max}"
        )
    return "\n".join(rows)


def from_meta(meta: dict[str, Any], spec: str | None) -> List[Bucket]:
    return build(
        parse(spec),
        max_audio_s=float(meta.get("max_audio_s", 25.0)),
        geometry=FrameGeometry.from_meta(meta),
    )


if __name__ == "__main__":
    print(describe(build(DEFAULT_BUCKETS_S)))
