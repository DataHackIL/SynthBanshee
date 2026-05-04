"""Integration test: real Google Cloud TTS round-trip (issue #40).

Verifies that the SSML produced by ``SSMLBuilder(supports_style_tags=False)``
is accepted by Google's API, and that ``GoogleProvider._pcm_to_wav``
produces audio that round-trips through ``soundfile`` as a 24 kHz mono
16-bit PCM WAV with non-silent content.

The mocked unit tests in ``tests/unit/test_google_provider.py`` cover
parsing, dispatch, and error handling; this test covers the contract
between our SSML and the actual Google service.

Marked with ``@pytest.mark.live_tts`` so it can be skipped in fast loops:

    pytest -m "not live_tts"      # skip live API calls
    pytest tests/integration/test_google_tts.py   # run only this test

Skipped automatically when ``GOOGLE_APPLICATION_CREDENTIALS`` is unset
or the optional ``google-cloud-texttospeech`` extra is not installed.
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import pytest
import soundfile as sf

from synthbanshee.tts.google_provider import GoogleProvider
from synthbanshee.tts.ssml_builder import SSMLBuilder

# Short Hebrew utterance — kept brief to keep the test cheap.
_HEBREW_UTTERANCE = "שלום, זאת בדיקה."

# Real Google he-IL Chirp 3 HD voice — verify with scripts/check_google_voices.py.
_VOICE_ID = "he-IL-Chirp3-HD-Achird"

# Google TTS LINEAR16 output rate, wrapped by GoogleProvider._pcm_to_wav.
_EXPECTED_SAMPLE_RATE_HZ = 24_000

# Floor for "clearly non-silent" speech on a [-1, 1] float32 scale.
_RMS_FLOOR = 0.005


def _credentials_available() -> bool:
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def _google_sdk_available() -> bool:
    try:
        import google.cloud.texttospeech  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


_SKIP_REASON = (
    "Live Google TTS test: requires GOOGLE_APPLICATION_CREDENTIALS and the "
    "google-cloud-texttospeech extra (`pip install 'synthbanshee[google-tts]'`)."
)


@pytest.mark.live_tts
@pytest.mark.skipif(
    not (_credentials_available() and _google_sdk_available()),
    reason=_SKIP_REASON,
)
def test_google_tts_round_trip_returns_valid_non_silent_wav() -> None:
    """One short Hebrew utterance survives the SSMLBuilder → Google → WAV path."""
    builder = SSMLBuilder()
    ssml = builder.build_from_speaker_config(
        text=_HEBREW_UTTERANCE,
        voice_id=_VOICE_ID,
        style="General",
        rate_multiplier=1.0,
        supports_style_tags=False,
    )

    # Regression guard: SSML for Google must never carry Azure-only mstts tags.
    assert "mstts" not in ssml
    assert "express-as" not in ssml

    wav_bytes = GoogleProvider().synthesize(ssml)

    assert wav_bytes[:4] == b"RIFF", "Output must start with a RIFF/WAV header"
    with wave.open(io.BytesIO(wav_bytes), "r") as w:
        assert w.getnchannels() == 1, "Output must be mono"
        assert w.getsampwidth() == 2, "Output must be 16-bit (sampwidth=2)"
        assert w.getframerate() == _EXPECTED_SAMPLE_RATE_HZ
        assert w.getnframes() > 0

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    assert sample_rate == _EXPECTED_SAMPLE_RATE_HZ
    assert audio.ndim == 1, "Output must be mono (single channel)"

    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > _RMS_FLOOR, f"Audio appears silent (RMS={rms:.5f})"


def test_ssml_builder_google_mode_emits_no_mstts() -> None:
    """Cheap regression guard for the ``supports_style_tags=False`` path.

    Runs without API access so it always executes — the round-trip test
    is gated on credentials, but this snapshot check is not.
    """
    builder = SSMLBuilder()
    ssml = builder.build_from_speaker_config(
        text=_HEBREW_UTTERANCE,
        voice_id=_VOICE_ID,
        style="angry",  # non-General style still must not produce express-as
        rate_multiplier=1.1,
        pitch_delta_st=2.0,
        supports_style_tags=False,
    )
    assert "mstts" not in ssml
    assert "express-as" not in ssml
    assert _VOICE_ID in ssml
