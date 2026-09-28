"""Vendored copy of ``gigaam.decoding.Tokenizer``.

Not imported from gigaam on purpose: ``gigaam/__init__.py`` reaches ``model.py``
and ``decoding.py``, both of which import torch at module level. The runtime
image ships without torch (spec §4.2), so importing the package at all would
drag ~2.5 GB back in for thirty lines of vocabulary lookup.

Kept deliberately faithful to upstream -- if that file changes, this one is the
thing to re-check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class Tokenizer:
    """Character-wise vocabulary or a SentencePiece model.

    Which one is in play depends on the variant: the plain ``v3_ctc`` / ``v3_rnnt``
    models are char-wise, the ``e2e`` ones (cased, punctuated) ship a
    SentencePiece model.
    """

    def __init__(self, vocab: Optional[List[str]] = None, model_path: Optional[str] = None):
        self.charwise = model_path is None
        if self.charwise:
            if not vocab:
                raise ValueError("char-wise tokenizer needs a vocabulary")
            self.vocab = list(vocab)
        else:
            from sentencepiece import SentencePieceProcessor

            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: List[int]) -> str:
        if self.charwise:
            return "".join(self.vocab[t] for t in tokens)
        return self.model.decode(tokens)

    def id_to_str(self, token_id: int) -> str:
        if self.charwise:
            return self.vocab[token_id]
        return self.model.IdToPiece(token_id)

    def __len__(self) -> int:
        return len(self.vocab) if self.charwise else len(self.model)

    @property
    def blank_id(self) -> int:
        """GigaAM puts the CTC blank one past the end of the vocabulary."""
        return len(self)

    @classmethod
    def from_dir(cls, model_dir: str) -> "Tokenizer":
        """Load from the artefacts the builder writes next to the model.

        Expects ``meta.json`` and, depending on the variant, either
        ``vocab.json`` or ``<name>_tokenizer.model``.
        """
        root = Path(model_dir)
        meta = json.loads((root / "meta.json").read_text())

        sp_name = meta.get("tokenizer_model")
        if sp_name:
            sp_path = root / sp_name
            if not sp_path.exists():
                raise FileNotFoundError(f"tokenizer model missing: {sp_path}")
            return cls(model_path=str(sp_path))

        vocab_path = root / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocabulary missing: {vocab_path}")
        return cls(vocab=json.loads(vocab_path.read_text()))
