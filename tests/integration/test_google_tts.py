"""Integration test: real Google Cloud TTS round-trip (issue #40).

The unit tests in ``tests/unit/test_google_provider.py`` cover SSML shape,
provider dispatch, and error handling — all with a mocked client.  This
test fills the remaining gap: that the SSML our code generates is
actually accepted by Google's API and that synthesizing a real Hebrew
utterance returns plausible, non-silent audio.

Skip behavior:

- Skipped when ``google-cloud-texttospeech`` is not installed (mirrors
  the lazy-import guard in ``google_provider.py``).
- Skipped when ``GOOGLE_APPLICATION_CREDENTIALS`` is unset.

Marker filter:

    pytest -m "not live_tts"   # skip live API calls in fast loops
"""

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthbanshee.config.speaker_config import SpeakerConfig
from synthbanshee.tts.google_provider import GoogleProvider
from synthbanshee.tts.ssml_builder import SSMLBuilder

# A known Google-voiced speaker YAML.  Drives the test from real config so
# that voice deprecation in YAMLs (or removal of the Google provider from
# this speaker) breaks this test instead of silently passing against a
# stale hard-coded voice ID.
_GOOGLE_SPEAKER_YAML = (
    Path(__file__).parent.parent.parent / "configs" / "speakers" / "speaker_BEN_M_40-55_004.yaml"
)

# Short Hebrew utterance — kept brief to keep the test cheap.  ~4 words.
_HEBREW_UTTERANCE = "שלום, זאת בדיקה."

# Google TTS LINEAR16 output rate, wrapped by GoogleProvider._pcm_to_wav.
_EXPECTED_SAMPLE_RATE_HZ = 24_000

# Floor for "clearly non-silent" speech on a [-1, 1] float32 scale.
_RMS_FLOOR = 0.005

# Plausibility window for a ~4-word Hebrew utterance.  A degenerate response
# (silence sliver, error fallback, partial audio) falls outside this range.
_DURATION_MIN_S = 1.0
_DURATION_MAX_S = 15.0


def _credentials_available() -> bool:
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def _google_sdk_installed() -> bool:
    """True iff ``google-cloud-texttospeech`` is importable.

    Uses ``find_spec`` so we don't trigger the SDK's import side effects
    at pytest collection time — same reason ``GoogleProvider`` lazy-imports.
    """
    return importlib.util.find_spec("google.cloud.texttospeech") is not None


@pytest.mark.live_tts
@pytest.mark.skipif(
    not _google_sdk_installed(),
    reason="google-cloud-texttospeech not installed (`pip install 'synthbanshee[google-tts]'`)",
)
@pytest.mark.skipif(
    not _credentials_available(),
    reason="GOOGLE_APPLICATION_CREDENTIALS is not set",
)
def test_google_tts_round_trip_returns_plausible_non_silent_audio() -> None:
    """A short Hebrew utterance synthesizes to plausible, non-silent audio.

    Asserts only what Google actually controls: the call succeeds, the
    returned audio is decodable as a 24 kHz mono float waveform, its
    duration is plausible for the input, and it is not silent.  WAV
    container shape is owned by ``_pcm_to_wav`` and covered by unit tests
    — not re-asserted here.
    """
    speaker = SpeakerConfig.from_yaml(_GOOGLE_SPEAKER_YAML)
    assert speaker.tts_provider == "google", (
        f"Test fixture speaker must use Google TTS, got {speaker.tts_provider!r}"
    )
    style_entry = speaker.style_for_intensity(2)

    builder = SSMLBuilder()
    ssml = builder.build_from_speaker_config(
        text=_HEBREW_UTTERANCE,
        voice_id=speaker.tts_voice_id,
        style=style_entry.style,
        rate_multiplier=style_entry.rate_multiplier,
        pitch_delta_st=style_entry.pitch_delta_st,
        volume_delta_db=style_entry.volume_delta_db,
        supports_style_tags=False,
    )

    wav_bytes = GoogleProvider().synthesize(ssml)
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    assert sample_rate == _EXPECTED_SAMPLE_RATE_HZ
    assert audio.ndim == 1, "expected mono audio"

    duration_s = audio.shape[0] / sample_rate
    assert _DURATION_MIN_S < duration_s < _DURATION_MAX_S, (
        f"audio duration {duration_s:.3f}s outside plausible window "
        f"[{_DURATION_MIN_S}, {_DURATION_MAX_S}] for utterance {_HEBREW_UTTERANCE!r}"
    )

    rms = float(np.sqrt(np.mean(audio**2)))
    assert rms > _RMS_FLOOR, f"audio appears silent (RMS={rms:.5f})"
