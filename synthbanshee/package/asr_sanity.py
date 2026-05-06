"""Whisper-based per-clip ASR sanity check (#87).

Detects the Whisper "backdoor" failure mode: clips that synthesize fine to a
listener but trip Whisper-large-v3's silence-detection / segmentation
heuristic, causing the encoder to mis-segment audio chunks as silence and
drop ~20–30% of words from the hypothesis.  This manifests as:

  - High WER (≥ ~0.20) without any obvious audio defect.
  - **Length ratio** (`hyp_words / ref_words`) collapsing well below 1.0
    (the #87 cases sat at 0.70–0.76; the 04-15 baseline was 1.00).

The length-ratio collapse is the canonical fingerprint and the metric that
``run_asr_sanity`` flags on.  WER alone is noisier (it counts substitutions
and insertions too) and produces false positives on natural pronunciation
variation.

Heavy dependencies (``transformers``, ``torch``, ``jiwer``) are intentionally
**lazy-imported** inside the runner so importing this module is cheap and
does not pull the Whisper weights into memory.  Install with:

    uv pip install --python .venv/bin/python -e ".[eval-asr]"

This module is invoked from ``qa-report --asr``; do not use it as a quality
gate during generation (per-clip Whisper inference is too expensive).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# Default Whisper backbone.  Pinned to large-v3 because the #87 evidence base
# was collected on this exact model; switching alters the silence-detection
# behaviour and would require re-baselining every flagged clip.
DEFAULT_MODEL = "openai/whisper-large-v3"
DEFAULT_LANGUAGE = "he"


# Regex factored out from `scripts/m17_phase_a_validation.py` so qa-report
# and the spike script agree on what counts as a "word" for WER.  Strip
# punctuation, normalize whitespace, leave Hebrew letters and digits intact.
_WER_PUNCT_RE = re.compile(r"[^\w\s֐-׿]+", re.UNICODE)
_WER_WS_RE = re.compile(r"\s+")


def normalize_for_wer(text: str) -> str:
    """Strip punctuation and collapse whitespace for jiwer's word tokenizer."""
    return _WER_WS_RE.sub(" ", _WER_PUNCT_RE.sub(" ", text)).strip()


def _read_reference_text(txt_path: Path) -> str:
    """Read the bracket-stripped Hebrew transcript from a clip's .txt file."""
    raw = txt_path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("[")]
    return "\n".join(lines).strip()


@dataclass(frozen=True)
class AsrMetrics:
    """Per-clip ASR sanity result."""

    wer: float
    length_ratio: float  # hyp_words / max(ref_words, 1)
    hyp_words: int
    ref_words: int
    hyp_text: str

    def is_silence_collapse(self, min_length_ratio: float) -> bool:
        """True when ``length_ratio`` is below the threshold (the #87 fingerprint)."""
        return self.length_ratio < min_length_ratio


class WhisperRunner:
    """Lazy wrapper around the HuggingFace Whisper ASR pipeline.

    Instantiating this class does NOT load the model — that happens on the
    first ``transcribe`` call.  Callers that never invoke ``transcribe`` pay
    nothing.  Re-using one instance across many clips amortizes the
    ~5–10 s model-load cost.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        chunk_length_s: int = 30,
    ) -> None:
        self.model = model
        self.language = language
        self.chunk_length_s = chunk_length_s
        # Typed as Any because the ASR pipeline subclass varies across
        # transformers versions and we lazy-import to avoid the dependency.
        self._asr: Any = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._asr is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:  # pragma: no cover - import-time error path
            raise ImportError(
                "ASR sanity requires the 'eval-asr' extra. Install with:\n"
                '  uv pip install -e ".[eval-asr]"\n'
                f"Underlying error: {e}"
            ) from e

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _log.info("Loading Whisper model %s on %s", self.model, device)
        self._asr = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=device,
            torch_dtype=torch.float32,
            chunk_length_s=self.chunk_length_s,
        )

    def transcribe(self, wav_samples, sample_rate: int) -> str:
        """Run the model on a numpy float32 mono buffer; return the hypothesis text."""
        self._ensure_loaded()
        assert self._asr is not None
        out = self._asr(
            {"raw": wav_samples.copy(), "sampling_rate": sample_rate},
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
                "num_beams": 1,
                "do_sample": False,
            },
        )
        return out["text"]


def compute_asr_metrics(
    wav_path: Path,
    txt_path: Path,
    runner: WhisperRunner,
) -> AsrMetrics:
    """Read a clip's WAV + reference TXT and compute WER + length-ratio.

    Reuses ``runner`` so the Whisper model is loaded at most once across many
    clips.  ``txt_path`` is parsed with ``_read_reference_text`` (bracket lines
    stripped); the resulting text is the WER reference.
    """
    try:
        from jiwer import wer
    except ImportError as e:  # pragma: no cover - import-time error path
        raise ImportError(
            "ASR sanity requires 'jiwer'. Install with: uv pip install -e \".[eval-asr]\""
        ) from e
    import soundfile as sf

    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    ref_text = _read_reference_text(txt_path)
    hyp_text = runner.transcribe(samples, sr)

    ref_norm = normalize_for_wer(ref_text)
    hyp_norm = normalize_for_wer(hyp_text)
    ref_words = max(len(ref_norm.split()), 1)
    hyp_words = len(hyp_norm.split())

    return AsrMetrics(
        wer=float(wer(ref_norm, hyp_norm)),
        length_ratio=hyp_words / ref_words,
        hyp_words=hyp_words,
        ref_words=ref_words,
        hyp_text=hyp_text,
    )
