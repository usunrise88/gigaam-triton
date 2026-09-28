"""Greedy CTC collapse over pre-argmaxed frames.

The argmax itself happens on the GPU inside the exported graph, not here. That
is not a style choice: for a 20 s bucket the log-prob tensor is
frames x vocab, and hauling that to a CPU Python backend just to take an argmax
would cost more than the whole encoder pass we are trying to keep under 25 ms.
The graph emits the argmax and its value per frame; this module only does the
cheap part -- collapse repeats, drop blanks, gather scores.

Full log-probs still leave the graph as a separate output for the provider to
decode itself (spec §7.2); they simply do not pass through Python.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def collapse(
    token_ids: np.ndarray,
    frame_scores: np.ndarray,
    length: int,
    blank_id: int,
) -> Tuple[List[int], List[float]]:
    """Standard CTC collapse for one sample.

    Returns emitted token ids and, for each, the model's log-probability at the
    frame it was emitted on. Confidence downstream is the mean of these.
    """
    n = max(0, min(int(length), int(token_ids.shape[0])))
    if n == 0:
        return [], []

    ids = token_ids[:n]
    scores = frame_scores[:n]

    keep = ids != blank_id
    keep[1:] &= ids[1:] != ids[:-1]

    return ids[keep].astype(np.int64).tolist(), scores[keep].astype(np.float64).tolist()


def decode(
    token_ids: np.ndarray,
    frame_scores: np.ndarray,
    length: int,
    blank_id: int,
    tokenizer,
) -> Tuple[str, List[int], List[float]]:
    tokens, scores = collapse(token_ids, frame_scores, length, blank_id)
    return tokenizer.decode(tokens), tokens, scores
