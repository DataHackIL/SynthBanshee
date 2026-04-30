"""List available Google Cloud TTS he-IL voices, highlighting Chirp 3 HD variants.

Usage:
    python scripts/check_google_voices.py

Requirements:
    GOOGLE_APPLICATION_CREDENTIALS must point to a valid service-account JSON.
    Install the client library:  pip install 'synthbanshee[google-tts]'
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        sys.exit(
            "Error: GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "Point it to a service-account JSON file and re-run."
        )

    try:
        from google.cloud import texttospeech
    except ImportError:
        sys.exit(
            "Error: google-cloud-texttospeech is not installed.\n"
            "Run:  pip install 'synthbanshee[google-tts]'"
        )

    client = texttospeech.TextToSpeechClient()
    response = client.list_voices(language_code="he-IL")

    all_voices = sorted(response.voices, key=lambda v: v.name)
    chirp_voices = [v for v in all_voices if "Chirp" in v.name]
    other_voices = [v for v in all_voices if "Chirp" not in v.name]

    gender_label = {
        texttospeech.SsmlVoiceGender.MALE: "MALE",
        texttospeech.SsmlVoiceGender.FEMALE: "FEMALE",
        texttospeech.SsmlVoiceGender.NEUTRAL: "NEUTRAL",
    }

    def print_voices(voices: list) -> None:
        for v in voices:
            gender = gender_label.get(v.ssml_gender, "UNKNOWN")
            print(f"  {v.name:<40}  {gender:<8}  {v.natural_sample_rate_hertz} Hz")

    print(f"\n{'=' * 70}")
    print("Google Cloud TTS — he-IL voices")
    print(f"{'=' * 70}")

    if chirp_voices:
        print(f"\n--- Chirp variants ({len(chirp_voices)}) ---")
        print_voices(chirp_voices)
    else:
        print("\n[!] No Chirp voices found for he-IL.")

    if other_voices:
        print(f"\n--- Other he-IL voices ({len(other_voices)}) ---")
        print_voices(other_voices)

    print(f"\nTotal: {len(all_voices)} voice(s)\n")

    if chirp_voices:
        males = [v.name for v in chirp_voices if v.ssml_gender == texttospeech.SsmlVoiceGender.MALE]
        females = [
            v.name for v in chirp_voices if v.ssml_gender == texttospeech.SsmlVoiceGender.FEMALE
        ]
        print("Speaker YAML values to use:")
        print(f"  Male   tts_voice_id: {males[0] if males else '(none found)'}")
        print(f"  Female tts_voice_id: {females[0] if females else '(none found)'}")
        print()


if __name__ == "__main__":
    main()
