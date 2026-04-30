# Research Report: Realistic Synthetic Hebrew Dialogue Audio Generation

**Target pipeline:** multi-speaker Hebrew dialogue clips, 2-5 minutes each, cloud TTS with SSML prosody control, post-processed to 16 kHz mono 16-bit PCM WAV for audio-classification training.
**Model/file:** GPT-5.5 Pro, `research_report_gpt55_pro.md`.
**Date:** 2026-04-30. Platform capability notes are time-sensitive and should be rechecked before production use.

## Executive Summary

The current pipeline has four interacting failure modes:

1. **The signal chain is damaging already-clean TTS.** Wiener denoising on clean synthetic speech, repeated filtering, hard joins, and possible over-normalization can remove the high-frequency consonant detail and micro-variation that make speech intelligible and alive. A 7.5 kHz low-pass at 16 kHz sampling rate is not by itself usually enough to create severe muffling because the Nyquist frequency is only 8 kHz, but a steep filter, a filter with passband ripple, denoising, codec simulation, or a chain of multiple low-pass/EQ operations can make the result muddy.
2. **The prosody model is probably being driven too directly.** Human emotion is not “anger = higher pitch.” Hot anger often has higher F0, wider F0 range, more intensity, faster articulation, tenser voice, and more high-frequency energy; cold anger may have lower or stable mean F0 with sharper timing and tense/creaky voice. For male speakers, anger can sound more realistic if the **range and intensity** increase while the median F0 remains near the speaker’s baseline.
3. **Cloud TTS controls are advisory, not deterministic DSP.** SSML `pitch`, `rate`, `volume`, and `mstts:express-as` are conditioning hints interpreted by the synthesis model. Unsupported styles can be ignored; excessive style intensity can shift timbre and apparent identity; combining pitch, rate, volume, and style tags can produce nonlinear artifacts.
4. **The result lacks a plausible acoustic scene.** Real phone-on-table audio includes room impulse response, early reflections, boundary effects, microphone coloration, automatic gain control, quiet nonstationary ambience, and sometimes codec artifacts. These should be added after TTS as a controlled augmentation stage, not by damaging the clean TTS before room/device simulation.

The highest-leverage changes are: remove Wiener denoising from clean TTS; synthesize each turn independently; use mild SSML; add quality gates for F0 drift, sustained vowels, clicks, high-frequency loss, and speaker-embedding consistency; then add room/device/noise simulation with realistic parameter distributions.

---

## Part 1: How Humans Perceive Speech Naturalness

### What Acoustic Features Make Speech Sound Real vs. Synthetic?

Human listeners judge speech naturalness from several layers at once:

#### Segmental accuracy

This is the correctness of phones and transitions: vowels, consonants, coarticulation, plosives, fricatives, and formant transitions. Even if the text is intelligible, listeners notice unnatural consonant releases, over-smoothed diphones, missing breath/noise components, or wrong stress placement.

For Hebrew, segmental errors can include incorrect realization of guttural/uvular consonants, unnatural /r/ quality, over-English stress timing, or unnatural treatment of final stress. Modern Hebrew TTS also needs correct handling of unvocalized text, heteronyms, acronyms, numbers, names, and clitic-like function words.

#### Prosodic plausibility

Prosody includes F0 contour, timing, phrase breaks, stress, rhythm, and loudness. Natural speech is neither perfectly smooth nor random. It has structured variability:

- F0 rises and falls at phrase boundaries.
- Stressed syllables are lengthened and/or intensified.
- Turn-final intonation depends on pragmatic function: question, continuation, resignation, accusation, contradiction.
- Speaking rate changes locally, not only globally.
- Emotional intensity changes pitch range, timing, spectral balance, voice quality, and articulation.

The synthetic failure pattern is often “too regular”: similar pause lengths, similar sentence-final falls, globally slow rate, and smoothed F0 curves without phrase-level intent.

#### Voice-source realism

Human voicing is produced by vocal-fold vibration plus turbulence and vocal-tract filtering. Real voices have small, structured irregularities. Relevant acoustic correlates include:

- **F0:** fundamental frequency, perceived as pitch.
- **F0 contour:** the time-varying pitch curve.
- **F0 range / SD:** how much pitch varies.
- **Jitter:** cycle-to-cycle F0 perturbation.
- **Shimmer:** cycle-to-cycle amplitude perturbation.
- **Harmonics-to-noise ratio (HNR):** harmonic energy relative to noise.
- **Breathiness:** turbulent glottal airflow; often higher aspiration noise and steeper spectral tilt.
- **Creakiness / laryngealization:** irregular low-frequency phonation, often with low F0 and reduced harmonic clarity.
- **Spectral tilt:** relative high-frequency energy versus low-frequency energy; tenser/louder/shouted speech tends to have flatter tilt and more high-frequency energy.

Classic voice-quality work by Klatt & Klatt (1990) showed that breathiness is not a single cue: listeners respond to combinations of spectral tilt, open quotient, aspiration noise, and source characteristics. Gobl & Ní Chasaide (2003) showed that voice qualities such as harsh, tense, breathy, whispery, creaky, and lax-creaky map strongly to perceived affect. These findings matter for TTS because “emotion” should not be implemented only as pitch and speed.

References:

- Klatt, D. H., & Klatt, L. C. (1990). *Analysis, synthesis, and perception of voice quality variations among female and male talkers*. Journal of the Acoustical Society of America. https://pubmed.ncbi.nlm.nih.gov/2231467/
- Gobl, C., & Ní Chasaide, A. (2003). *The role of voice quality in communicating emotion, mood and attitude*. Speech Communication. https://www.cs.columbia.edu/~julia/papers/gobl03.pdf
- Kreiman, J., & Gerratt, B. R. (2000s work on voice quality perception and acoustic correlates). Useful background: https://www.phonetics.ucla.edu/voice/

#### Environmental plausibility

Real recordings have an acoustic context. A clean studio TTS file rendered into 16 kHz PCM, with no room, no microphone coloration, no handling noise, no device AGC, and no subtle ambience, sounds like a TTS sample even if the voice model is good. In a real phone-on-table recording, the listener hears:

- direct speech plus room reflections;
- microphone off-axis coloration;
- table boundary effects;
- low-level room tone;
- changing source-to-mic distance as people move;
- automatic gain control and compression;
- small bursts of non-speech sound;
- possible codec or packet artifacts if captured through telephony.

Naturalness is therefore a property of the **whole scene**, not just the speech waveform.

### Psychoacoustic Research on Speech Naturalness Perception

#### F0 dynamics and emotion perception

Scherer’s work on vocal communication of emotion is central. Banse & Scherer (1996) analyzed 224 acted emotional portrayals and found that vocal parameters differentiate not just emotional intensity but also emotion type. Scherer (2003) reviewed evidence that listeners recognize anger and sadness relatively well from voice, while emotions such as disgust are more difficult vocally.

Key sources:

- Banse, R., & Scherer, K. R. (1996). *Acoustic profiles in vocal emotion expression*. Journal of Personality and Social Psychology. https://pubmed.ncbi.nlm.nih.gov/8851745/ and PDF mirror: https://www.columbia.edu/~rmk7/HC/HC_Readings/Scherer.pdf
- Scherer, K. R. (2003). *Vocal communication of emotion: A review of research paradigms*. Speech Communication. https://www1.cs.columbia.edu/~julia/papers/2003_Scherer_SpeechComm.pdf
- Bänziger, T., & Scherer, K. R. (2005/2010 GEMEP-related work). The Geneva Multimodal Emotion Portrayals corpus is relevant for acted emotional speech.

The practical lesson: F0 mean alone is insufficient. A realistic emotional utterance should combine F0 level, F0 range, intensity, rate, pausing, spectral tilt, articulation precision, and voice quality.

#### Voice quality and perceived affect

Listeners infer affect from voice-source quality. Gobl & Ní Chasaide (2003) found that different synthetic voice qualities trigger different perceived emotional/mood attributes. Harsh/tense voice is associated with anger or effort; breathy/whispery voice can signal fear, intimacy, weakness, or distress depending on context; creaky/lax-creaky voice can signal sadness, fatigue, resignation, or authority depending on language and speaker.

For synthetic dialogue, voice quality should be applied sparingly and contextually. Adding global jitter or breath noise across an entire clip can make the output sound pathological. Add micro-prosody locally and conditionally: phrase-final creak, breathiness under distress/fear, tense phonation during anger, and vocal effort during shouting.

#### Timing and rhythm

Duration is a powerful naturalness cue. Overly even timing is a common synthetic artifact. Natural speech has phrase-level acceleration/deceleration, local lengthening before boundaries, and variable pause durations. Speech rate also differs between **articulation rate** and **overall rate**:

- Articulation rate excludes pauses.
- Overall speaking rate includes pauses.

A clip can have a fast articulation rate and still feel slow if pauses are too long or too regular. Conversely, a low-intensity calm voice can be natural with slow articulation if pause placement is meaningful.

#### Spectral balance and clarity

Speech intelligibility depends heavily on the 1-5 kHz region, especially consonants. At 16 kHz sampling rate, the usable bandwidth reaches 8 kHz. Energy above 4 kHz contributes fricatives, brightness, air, and microphone realism. Excess attenuation in 3-8 kHz makes speech muffled; excess low-mid energy around 150-500 Hz makes speech muddy; aggressive denoising can create musical noise and remove fricative detail.

### F0, Jitter, Shimmer, Breathiness, and Spectral Tilt

#### F0 level vs. F0 contour

F0 level is the speaker’s median or mean pitch. F0 contour is the time-varying pitch movement. Human naturalness depends more on the contour being pragmatically plausible than on mean F0 matching a target.

Typical adult conversational F0 ranges used as engineering references:

| Speaker class | Typical mean/median F0 | Common conversational range | Notes |
|---|---:|---:|---|
| Adult male | ~100-135 Hz | ~80-180 Hz | Peaks above 180-220 Hz can occur but should not dominate long male dialogue unless the speaker is unusually high-pitched or highly aroused. |
| Adult female | ~170-230 Hz | ~140-320 Hz | Sustained medians below ~150 Hz may sound unusually low for many adult female voices unless the voice identity supports it. |

These are general voice-science ranges, not Hebrew-specific norms. A compact reference is Pépiot (2014), who reports typical mean F0 values around 120 Hz for men and 200 Hz for women in a cross-language gender-voice discussion. Hebrew-specific normative conversational F0 distributions are not as easy to find in open, production-ready form; for a Hebrew dataset, estimate per-speaker baselines empirically from Hebrew corpora and your generated voices.

Reference:

- Pépiot, E. (2014). *Male and female speech: A study of mean f0, f0 range, phonation type and speech rate in Parisian French and American English speakers*. https://www.isca-archive.org/speechprosody_2014/pepiot14_speechprosody.html

#### Jitter and shimmer

Clinical voice-analysis references often use sustained vowels, not conversational speech. Common rough thresholds for healthy sustained phonation are:

- jitter: usually below ~0.5-1.0%;
- shimmer: usually below ~3-5%;
- HNR: often above ~20 dB for healthy modal vowels.

These should not be treated as frame-level targets for continuous speech. Connected speech naturally contains transitions, devoicing, consonantal noise, and prosodic changes that make jitter/shimmer estimates less stable. In synthetic augmentation, do not add independent random jitter/shimmer to the waveform. Use controlled, low-amplitude F0 and amplitude perturbations only on voiced regions, preferably through a source-filter/vocoder representation.

Practical targets for post-synthesis micro-variation:

- F0 micro-perturbation: 0.1-0.5% RMS within voiced vowels, correlated over 20-80 ms.
- Shimmer-like amplitude perturbation: 0.3-1.5% RMS, correlated over 20-80 ms.
- Phrase-final creak: occasional, not global; 80-300 ms at final syllables.
- Breath noise: only on breathy/distressed/fearful vowels or at inhale/exhale events; keep 6-20 dB below voiced harmonic energy depending on intended audibility.

#### Breathiness

Breathiness involves aspiration noise and a glottal configuration that changes the harmonic spectrum. Naively mixing white noise into speech sounds like hiss, not breathy phonation. More realistic breathiness uses shaped noise concentrated in the formant bands and coupled to voiced vowels.

Engineering approximation:

- detect voiced vowel-like regions;
- synthesize pink/brownish or high-passed aspiration noise;
- filter it through broad vowel/formant bands or use LPC residual shaping;
- mix at -20 to -10 dB relative to local speech RMS for subtle breathiness; -10 to -6 dB for obvious distress/whisper effects;
- never apply across fricatives/plosives indiscriminately.

#### Spectral tilt

Spectral tilt is the slope of the harmonic spectrum. Loud, tense, or shouted speech usually has flatter tilt, more high-frequency energy, stronger glottal closure, and more intensity. Breathy speech often has steeper tilt and more turbulent noise. Creaky voice often has irregular periodicity, low F0, and distinctive low-frequency emphasis.

Useful spectral features:

- H1-H2: first harmonic minus second harmonic; related to open quotient/breathiness.
- H1-A3: first harmonic minus third formant region; often used as a voice-quality cue.
- Alpha ratio: energy above vs. below a cutoff, often around 1 kHz.
- High-frequency energy ratio: e.g., 4-8 kHz / 0-8 kHz for 16 kHz audio.
- Spectral centroid and spectral rolloff.

### How Emotion Manifests Acoustically in Real Speech

No single acoustic recipe maps cleanly to an emotion. Emotion expression varies by speaker, language, acting vs. spontaneous speech, intensity, social context, and recording condition. The values below are practical engineering ranges derived from emotion-phonetics literature and common adult voice ranges; they should be calibrated on real Hebrew recordings when possible.

#### General pattern from phonetics literature

Banse & Scherer (1996), Paeschke & Sendlmeier (2000), Laukka (2004), and Scherer (2003) agree on several broad patterns:

- High-arousal emotions such as hot anger and fear tend to increase F0 level and F0 variability.
- Anger and fear can both raise F0 and rate; they differ in voice quality, articulation, spectral energy, and contour details.
- Hot anger often has high intensity, fast articulation, precise/forceful articulation, flatter spectral tilt, and more high-frequency energy.
- Fear often has high F0, wide F0 range, breathiness/tense-breathy quality, and sometimes irregular timing.
- Sadness/distress often has slower rate, lower intensity, longer pauses, lower or unstable F0, breathy/creaky quality, and narrower or broken contours.
- Calm speech typically has moderate-low intensity, smoother contours, less extreme spectral energy, and longer pauses.

References:

- Paeschke, A., & Sendlmeier, W. F. (2000). *Prosodic characteristics of emotional speech: Measurements of fundamental frequency movements*. https://www.isca-archive.org/speechemotion_2000/paeschke00_speechemotion.html
- Laukka, P. (2004). *Vocal Expression of Emotion*. https://www.diva-portal.org/smash/get/diva2:165425/fulltext01.pdf
- Scherer, K. R. (2003). *Vocal communication of emotion*. https://www1.cs.columbia.edu/~julia/papers/2003_Scherer_SpeechComm.pdf

#### Practical acoustic target table

Use these as **relative** targets from a per-speaker neutral baseline, not as universal values.

| State | Mean/median F0 | F0 range/SD | Rate | Intensity | Spectral/voice quality | Engineering notes |
|---|---:|---:|---:|---:|---|---|
| Calm / neutral | baseline | baseline or -10% | baseline or -5-15% | baseline or -3 dB | modal, smooth, moderate tilt | Avoid making calm uniformly slow. Use meaningful pauses. |
| Hot anger | +0-25% for men, +5-35% for women; peaks can rise more | +30-100% | +10-35% articulation | +6-15 dB local peaks | tense/pressed, flatter tilt, stronger 2-5 kHz, sharper consonants | For male voices, prefer stable median + wider range instead of constant high pitch. |
| Cold anger | -10% to +10% | +10-40% or narrower but sharper accents | -5% to +15% | +0-8 dB, controlled | tense, creaky/pressed, clipped timing | Good for threats, sarcasm, controlled confrontation. |
| Fear / anxiety | +15-50% | +40-120% | +5-30%, sometimes broken | +0-10 dB | breathy-tense, tremor, high F0, irregular pauses | Add short inhalations, hesitations, incomplete clauses. |
| Distress / despair | -15% to +20%, unstable | +10-80%, often broken | -20% to +10% | -6 to +6 dB | breathy, creaky, crying-like breaks | Do not overuse low pitch; distress can include pitch spikes. |
| Shouting | speaker-dependent; not always higher median | very wide | fast bursts, clipped pauses | +10-20 dB before AGC | high-frequency energy, compression, saturation, vocal effort | Shouting is not just `volume="loud"`; use spectral and dynamic changes. |

For your reported problem, “male speaker pitch rises with anger instead of staying stable,” the fix is not to remove pitch variability. Instead:

- keep male median F0 within roughly 0-2 semitones of baseline for many angry turns;
- increase F0 range and accent peaks locally;
- increase intensity, articulation rate, consonant sharpness, and spectral brightness;
- shorten response latency and pauses;
- add tense voice quality or slight saturation after room/device simulation.

### Hebrew-Specific Prosody Patterns

#### Stress

Modern Hebrew has lexical stress, commonly final or penultimate depending on morphology and word class. Unlike English, F0 is not the strongest word-level stress cue in many Hebrew studies. Duration is especially important.

Silber-Varod et al. (2016), *The acoustic correlates of lexical stress in Israeli Hebrew*, report that vowel duration is the dominant stress marker, with intensity also contributing and F0 playing a relatively minor role. Their regression analysis attributed most explained variance to duration, with intensity and F0 contributing much less. This is important for SSML: do not try to make Hebrew stress natural by adding pitch bumps to every stressed syllable.

References:

- Silber-Varod, V., et al. (2016). *The acoustic correlates of lexical stress in Israeli Hebrew*. https://cris.tau.ac.il/en/publications/the-acoustic-correlates-of-lexical-stress-in-israeli-hebrew
- Bat-El, O., et al. (2019). Work on Modern Hebrew stress and duration. https://www.researchgate.net/publication/334736783_Stress_in_Modern_Hebrew

Engineering implications:

- Use phrase-level F0 contours rather than word-by-word pitch accents.
- For emphasis, lengthen the stressed vowel slightly and increase intensity modestly.
- Avoid English-like alternating stress rhythm.
- Pay attention to final-syllable lengthening at phrase boundaries.

#### Intonation

Mixdorff & Amir (2002), *The Prosody of Modern Hebrew*, describe common patterns such as falling contours in declaratives, rises for non-terminal/continuation accents, strong final rises in questions, and alignment of tone switches with accented syllables. Focus tends to boost local prominence.

Reference:

- Mixdorff, H., & Amir, N. (2002). *The prosody of Modern Hebrew*. https://www.isca-archive.org/speechprosody_2002/mixdorff02_speechprosody.html

Engineering implications:

- Declaratives should not all end with identical falls. Accusations, interruptions, and disbelief often use different contours.
- Continuation turns should preserve non-final rises or level endings.
- Yes/no questions can have strong terminal rises; content questions may not always use an English-like terminal rise.
- Hebrew dialogue needs turn-taking prosody: overlaps, quick replies, cutoffs, and backchannels.

#### Rhythm class

Hebrew rhythm is not simply English stress-timed rhythm. It has lexical stress but less pervasive vowel reduction than English, and many utterances feel more syllabically even. Rhythm-class metrics are debated across languages, and modern Hebrew should be treated empirically. For your pipeline, this means:

- avoid uniformly slow TTS;
- avoid English-style schwa reduction assumptions;
- make phrase timing depend on syntax and discourse intent;
- vary syllable durations around stress and boundaries, not evenly across the whole sentence.

### What Makes Recorded Phone Audio Different from Studio Audio?

#### Bandwidth

Common speech bandwidth categories:

| Category | Nominal bandwidth | Typical sample rate | Perceptual result |
|---|---:|---:|---|
| Narrowband telephony | ~300-3400 Hz | 8 kHz | telephone sound, limited brightness/fricatives, reduced naturalness |
| Wideband telephony | ~50-7000 Hz | 16 kHz | clearer speech, still not full studio quality |
| Super-wideband/fullband | up to 14-20 kHz | 32-48 kHz | more air/brightness, music-quality capture possible |

References:

- ITU-T G.722 wideband speech coding: 50-7000 Hz. https://www.itu.int/rec/T-REC-G.722
- ITU-T P.341 handset/headset transmission characteristics. https://www.itu.int/rec/T-REC-P.341
- AMR-WB background: https://www.loc.gov/preservation/digital/formats/fdd/fdd000108.shtml
- Opus RFC 6716: https://www.rfc-editor.org/rfc/rfc6716

For a 16 kHz WAV output, do not make every sample telephone-narrowband. If your downstream model will encounter mixed real-world audio, include a distribution:

- 60-80% wideband-like phone/table recordings, 80 Hz-7.2 kHz useful bandwidth;
- 10-25% narrowband/codec-heavy recordings, 300-3400 Hz;
- 10-20% cleaner near-field or headset-like recordings.

#### Frequency response and microphone coloration

Smartphone microphones are small electret/MEMS microphones with device-specific frequency responses. Phone software may apply:

- automatic gain control;
- noise suppression;
- beamforming;
- high-pass filtering;
- de-essing or spectral shaping;
- dynamic range compression.

A phone on a table also changes pickup geometry:

- boundary reflections from the table create comb filtering;
- the phone may point away from one speaker;
- low-frequency handling/table rumble can appear;
- high frequencies may attenuate off-axis.

#### Noise floor and ambience

Studio TTS has no room tone. Real phone audio has noise floor and nonstationary details: HVAC, refrigerator hum, distant traffic, chair movement, keyboard taps, cloth rustle, cutlery, footsteps, computer fans, and room reflections. A realistic scene should include low-level ambience even in “quiet” examples.

#### Codec artifacts

Phone-quality audio may contain:

- band limitation;
- quantization noise;
- pre-echo or smearing;
- packet loss concealment;
- robotic warble at low bitrates;
- clipping/limiting from AGC.

Use codec simulation sparingly unless the target deployment includes telephony. Excessive codec artifacts can train a classifier on codec cues rather than semantic/acoustic cues.

---

## Part 2: State of the Art in Neural TTS Naturalness

### Azure Neural TTS for Hebrew

Microsoft’s Azure AI Speech language support page lists Hebrew `he-IL` neural TTS voices including:

- `he-IL-AvriNeural` — male;
- `he-IL-HilaNeural` — female.

Official docs:

- Azure language and voice support: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
- Azure SSML voice customization: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
- Azure Text to Speech overview: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech

#### Strengths

- Reliable API and batch workflows.
- Good baseline intelligibility for Hebrew.
- Mature SSML support for standard neural voices.
- Supports prosody tags for pitch, rate, volume, and contour, subject to voice/model support.
- Supports audio effects such as `eq_telecomhp8k` for telecom-style output, though Microsoft recommends 8 kHz output for that effect.

#### Limitations for this use case

- As of the checked Microsoft docs, Hebrew Avri/Hila are listed as neural voices, but they do not appear in the documented table of voices with `mstts:express-as` styles/roles. Treat Hebrew emotional styles as unsupported unless your specific resource/region confirms otherwise.
- Standard SSML prosody can adjust delivery but does not create fully realistic spontaneous emotional acting.
- Extreme prosody values can be ignored or clipped internally. Microsoft explicitly states that prosody values are treated as suggestions and unsupported values can be substituted.
- Style, if supported for a voice, operates at sentence level and can change apparent timbre.
- Long requests increase risk of drift, repeated phrasing, and boundary issues.

### Google Cloud TTS Chirp 3 HD for Hebrew

Google Cloud’s Chirp 3 HD page lists Hebrew (Israel), `he-IL`, among supported languages, and describes Chirp 3 HD as a newer generation with 30 voices and more natural intonation/nuance. The same page lists LINEAR16 output support and describes preview SSML/voice controls for synchronous requests. Google’s broader “voices and types” page has historically stated that Chirp 3 HD does not support SSML input or pitch/rate audio parameters; the dedicated Chirp 3 HD page now lists preview support for some SSML tags and pace control. Treat this as version/region/API dependent and test the exact endpoint.

Official docs:

- Chirp 3 HD: https://cloud.google.com/text-to-speech/docs/chirp3-hd
- List of voice types: https://cloud.google.com/text-to-speech/docs/list-voices-and-types

#### Strengths

- Strong neural naturalness for many languages.
- Hebrew language support is explicitly listed for Chirp 3 HD.
- 30 named voices allow more speaker variety.
- LINEAR16 output is available.
- The Chirp 3 HD docs provide practical script-writing tips: punctuation matters, conversational writing helps, pauses/disfluencies can be scripted, and long complex sentences should be broken up.

#### Limitations

- Emotional control is indirect. There is no simple “anger intensity” control comparable to a true acted emotional model.
- Preview SSML support may ignore unsupported tags.
- Pace control may exist through `speaking_rate`, but pitch control may not be available for Chirp 3 HD depending on API/version.
- Output style can vary by voice; some voices are naturally warmer, lower, brighter, or more conversational.

### What SSML Prosody Parameters Actually Do

SSML controls do not usually apply deterministic DSP to a finished waveform. In neural TTS they are interpreted by the text-to-acoustic model as conditioning variables or hints. That has several consequences:

1. **Values are not exact.** `rate="+10%"` does not guarantee a measured 10% rate increase for every sentence.
2. **Controls interact.** Raising pitch can alter perceived age, gender, loudness, and emotion. Slowing rate can change pause placement and vowel duration. Raising volume may affect spectral balance or compression.
3. **Unsupported combinations can be ignored.** Azure docs state that unsupported prosody values can be substituted or ignored. Google Chirp 3 HD preview SSML may ignore unsupported tags.
4. **Style embeddings can shift identity.** If `express-as` is supported, the model may move in a learned style space where timbre, pitch, rate, and articulation are entangled.
5. **Sentence scope matters.** Azure documents `mstts:express-as` as a sentence-level adjustment. Applying style to tiny spans or individual words can behave differently across UI/API paths.

### Azure SSML Prosody Controls

Azure supports the SSML `<prosody>` element for many standard neural voices. Documented attributes include:

- `pitch` — baseline pitch. Microsoft recommends values within 0.5x to 1.5x of the original pitch; documented named values include `x-low`, `low`, `medium`, `high`, `x-high`, with approximate values such as `low=-20%`, `high=+20%`, `x-high=+45%`.
- `contour` — pitch targets at specified percentages of the utterance. Microsoft notes contour does not work well on single words or short phrases; it should be used on whole sentences or longer phrases.
- `range` — pitch range.
- `rate` — speaking rate. Documented range is 0.5x to 2x; named values include `x-slow`, `slow`, `medium`, `fast`, `x-fast`.
- `volume` — absolute value 0-100 or named values such as `silent`, `x-soft`, `soft`, `medium`, `loud`, `x-loud`.

For natural speech, use a much narrower range than the documented technical limits.

#### Safe Azure engineering ranges

| Parameter | Technical support may allow | Recommended natural range | Avoid |
|---|---:|---:|---|
| `rate` | 0.5x-2x | 0.90-1.12 for most turns; 0.85-1.18 for emotional extremes | Long spans below 0.80 or above 1.20 in Hebrew dialogue |
| `pitch` | named values or relative changes | -4% to +6% common; up to ±10% for short emotional spans | Sustained male anger at +15-25%; sustained female distress below -15% |
| `contour` | sentence-level | 2-4 control points per sentence | Word-level contouring; jagged contours |
| `volume` | 0-100 or named | `medium`, `loud`, or modest numeric changes | Using volume as a substitute for shouting |
| `break` | broad support | 80-450 ms conversational; 600-900 ms for dramatic pauses | Multi-second pauses inside TTS requests; insert long silence in post instead |
| `styledegree` | 0.01-2 for supported styles | 0.3-1.0, rarely 1.2 | >1.3 unless manually inspected |

### Known Issues with `mstts:express-as`

For your Hebrew pipeline, the most important point is that `mstts:express-as` is only supported by a subset of voices/styles. If a Hebrew voice does not list styles in Microsoft’s style table, do not depend on `style="angry"`, `style="cheerful"`, etc. Unsupported style use can be ignored or produce inconsistent behavior.

Known practical risks:

- **Voice identity shifts:** style embeddings can change timbre, apparent age, pitch, and articulation.
- **Style blending artifacts:** nesting or combining style with strong pitch/rate changes can produce unnatural hybrids.
- **Span-size sensitivity:** style applied to a single word may not behave like style applied to a sentence.
- **Role effects:** Microsoft documents roles as changing pitch and intonation to imitate age/gender while keeping the same voice name. This can be useful for role-play, but it is exactly the kind of control that can damage speaker identity consistency.

Recommendation: for Hebrew Avri/Hila, keep `express-as` off unless your tested endpoint explicitly supports it. Create emotional delivery with text, punctuation, mild prosody, multiple candidate synthesis, and post-processing.

### Known Issues with Combined Prosody Tags

Problematic combinations:

- high pitch + slow rate + loud volume: can sound childlike or strained;
- low pitch + slow rate: can make female voices unrealistically low or depressed;
- high rate + high pitch + short text: can produce clipped or chirpy output;
- many `<break>` tags: can sound like TTS reading markup rather than speaking;
- long SSML documents with many nested tags: greater chance of drift and artifacts;
- unsupported `express-as` + prosody: may be ignored or shift voice unpredictably.

For neural TTS, write the text as a human would say it first. Use SSML as a small correction layer.

### Best Practices for Natural-Sounding Emotional SSML

#### General principles

- Synthesize one speaker turn at a time, usually 2-15 seconds.
- Keep a fixed `voice name` per speaker identity.
- Do not switch voice/style mid-turn unless intentionally creating a character change.
- Avoid global rate reductions for calm speech; use local phrase pauses and softer intensity.
- Use punctuation and wording to express emotion before SSML.
- Generate multiple candidates and automatically score them.
- Prefer per-turn variation over one long monotonic escalation curve.

#### Azure Hebrew calm example

```xml
<speak version="1.0" xml:lang="he-IL"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="he-IL-HilaNeural">
    <prosody rate="-3%" pitch="+1%" volume="medium">
      אני מבינה. בוא נדבר על זה רגע, בלי למהר.
      <break time="260ms"/>
      מה בדיוק קרה שם?
    </prosody>
  </voice>
</speak>
```

#### Azure Hebrew controlled anger example

```xml
<speak version="1.0" xml:lang="he-IL"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="he-IL-AvriNeural">
    <prosody rate="+7%" pitch="-1%" volume="loud">
      לא. זה לא מה שסיכמנו.
      <break time="120ms"/>
      אמרת שתשלח את זה אתמול, ועכשיו אתה אומר לי שאין כלום?
    </prosody>
  </voice>
</speak>
```

Notes:

- The male pitch is kept stable/slightly lower.
- Anger is expressed through rate, wording, shorter pause, and loudness.
- Add room/device/saturation later; do not force all anger in TTS.

#### Azure supported-style example, only for voices that document the style

```xml
<speak version="1.0" xml:lang="en-US"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="en-US-SaraNeural">
    <mstts:express-as style="angry" styledegree="0.7">
      <prosody rate="+4%" pitch="-1%">
        That is not what we agreed on.
      </prosody>
    </mstts:express-as>
  </voice>
</speak>
```

Do not assume this works for Hebrew voices.

#### Google Chirp 3 HD example pattern

The exact API fields may differ by client library and preview status. Prefer API-level `speaking_rate` where supported and use `markup`/SSML only after endpoint testing.

```python
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

input_text = texttospeech.SynthesisInput(
    markup="""
    רגע. [pause short] אתה אומר לי שעוד לא התחלת?
    [pause short] אחרי כל השיחות שהיו לנו?
    """
)

voice = texttospeech.VoiceSelectionParams(
    language_code="he-IL",
    name="<tested-chirp3-hd-voice-name>",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    speaking_rate=1.06,
)

response = client.synthesize_speech(
    input=input_text,
    voice=voice,
    audio_config=audio_config,
)
```

### Professional TTS/SSML Tricks Used in Voice-Over and Audiobooks

Professional TTS workflows generally do not rely on a single pass.

Common tactics:

1. **Script rewriting for speech.** Replace written text with spoken text. Add contractions/equivalents, interruptions, discourse markers, hesitations, and shorter clauses.
2. **Punctuation as direction.** Commas, ellipses, question marks, dashes, and line breaks influence neural TTS more naturally than heavy SSML.
3. **Short renders.** Render paragraph/turn-sized segments, not full chapters/dialogues in one request.
4. **Multiple takes.** Generate several candidates per line and select by ASR alignment, duration, F0 statistics, MOS predictor, and human spot-checking.
5. **Phoneme and alias controls.** Use `<phoneme>`, `<say-as>`, or custom pronunciation dictionaries for names, acronyms, numbers, and Hebrew heteronyms where supported.
6. **Manual breath and pause placement.** Insert breaths and long silences in post, not only via SSML.
7. **Mastering as a separate stage.** Apply EQ, compression, room, device, and loudness after synthesis.
8. **Scene consistency.** Keep a stable room/device profile across a clip; do not randomize every turn independently unless the scene changes.

### Engine Comparison for Hebrew and Emotional Speech

| Engine | Hebrew support | Emotional control | Prosody control | Strengths | Risks / limitations |
|---|---|---|---|---|---|
| Azure Neural TTS | Yes: AvriNeural, HilaNeural | Limited for he-IL; styles appear unsupported in public style table | Mature SSML for standard voices | Reliable, stable, batch-friendly | Emotion may sound controlled/neutral; unsupported `express-as` can be ignored; style/timbre entanglement |
| Google Chirp 3 HD | Yes: `he-IL` listed | Indirect; voice/script driven | Pace and preview SSML depending on endpoint | Strong naturalness, many voices | Less deterministic; preview controls; pitch control may be unavailable |
| ElevenLabs | Hebrew advertised, model-dependent | Stronger expressive/voice-style capabilities | API/model dependent | Often high naturalness and emotional richness | Cost, licensing/privacy, less transparent controls, possible pronunciation issues; capabilities change frequently. Docs: https://elevenlabs.io/languages/hebrew |
| Coqui XTTS v2 | Official list does not include Hebrew | Voice cloning/style limited | Open-source controllability | Self-hostable; good for supported languages | Hebrew unsupported without custom training/fine-tuning. Docs: https://docs.coqui.ai/en/latest/models/xtts.html |
| Bark | Multilingual generative audio, expressive nonverbal sounds | Prompt-like, stochastic | Low deterministic control | Can generate laughs, sighs, ambience-like outputs | Hebrew reliability uncertain; artifacts; poor production determinism. GitHub: https://github.com/suno-ai/bark |
| HebTTS | Hebrew research/open TTS | Limited | Research-level | Hebrew-specific, diacritics-free work | Not necessarily production-natural. GitHub/paper: https://github.com/slp-rl/HebTTS |
| Mamre-TTS / Israwave | Hebrew-focused open projects | Limited | Project-specific | Useful experimentation | May not match cloud TTS naturalness; validate licensing and quality. Mamre: https://github.com/mamre-tts/mamre-tts |

Recommended strategy:

- Use Azure/Google as stable baselines.
- Test ElevenLabs for high-emotion scenes if licensing and data policy permit.
- Use open Hebrew TTS for diversity only after objective and human validation.
- Do not mix engines within a single speaker identity unless you intentionally want speaker changes.

---

## Part 3: Post-Processing for Realism

### Overall Post-Processing Philosophy

Do not degrade the TTS before it is intelligible and artifact-free. The recommended order is:

1. Generate clean dry TTS per turn.
2. Validate and reject broken turns.
3. Trim/crossfade boundaries.
4. Align turns on a dialogue timeline with overlaps and pauses.
5. Apply speaker positions and room impulse responses.
6. Apply device/microphone coloration and AGC/compression.
7. Mix ambience/noise/events at realistic SNR.
8. Optionally apply codec simulation.
9. Normalize/export to 16 kHz mono 16-bit PCM WAV.
10. Run final quality gates.

### Room Acoustics Simulation

#### Tools

- `pyroomacoustics`: Python package for room simulation, image-source method, beamforming, and RIR generation. Docs: https://pyroomacoustics.readthedocs.io/
- `gpuRIR`: GPU-accelerated RIR simulation using the image-source method. Useful when creating many source/receiver configurations. GitHub: https://github.com/DavidDiazGuerra/gpuRIR
- `rir-generator`: classic Python/MATLAB image-source RIR generation.
- Real RIR databases: MIT IR Survey, OpenAIR, EchoThief, BUT ReverbDB, OpenSLR RIRS_NOISES.

#### Image-source method vs. ray tracing

The image-source method, introduced by Allen & Berkley (1979), models specular reflections by mirroring sources across room boundaries. It is efficient and good for early reflections in rectangular rooms.

Ray tracing models many rays and can better approximate late reverberation, scattering, and complex geometries, but it is more expensive and stochastic.

Recommended approach:

- Use real RIRs when possible for final dataset realism.
- Use image-source simulation for controlled augmentation and easy parameter sweeps.
- Use hybrid ISM + ray tracing for larger rooms or when late reverberation matters.
- For a phone-on-table in small rooms, early reflections and direct-to-reverberant ratio matter more than long lush reverb.

Reference:

- Allen, J. B., & Berkley, D. A. (1979). *Image method for efficiently simulating small-room acoustics*. Journal of the Acoustical Society of America.
- pyroomacoustics docs: https://pyroomacoustics.readthedocs.io/

#### Realistic small-room parameter ranges

| Environment | Room size | RT60 | Source-mic distance | Notes |
|---|---:|---:|---:|---|
| Small bedroom | 3 x 3.5 x 2.6 m | 0.25-0.50 s | 0.7-2.5 m | Soft absorption, bed/curtains reduce reverb |
| Living room | 4 x 5 x 2.7 m | 0.35-0.70 s | 1-4 m | Common phone-on-table scene |
| Kitchen | 3 x 4 x 2.6 m | 0.45-0.85 s | 1-3 m | Hard surfaces, brighter reflections |
| Office | 3 x 4 x 2.7 m | 0.30-0.60 s | 0.6-2.5 m | HVAC/computer noise likely |
| Conference room | 4 x 7 x 2.8 m | 0.45-0.90 s | 1-5 m | More late reverb, table reflections |
| Bathroom/tiled room | 2 x 3 x 2.5 m | 0.80-1.50 s | 0.5-2 m | Use rarely unless target data includes it |

RT60 of 0.4-0.7 s is common for speech-oriented rooms; studios and treated booths are lower. Avoid applying a heavy cathedral-like reverb to all clips.

#### Direct-to-reverberant ratio

For a phone on a table:

- Near speaker at 0.5-1 m: DRR roughly +3 to +10 dB.
- Normal table conversation at 1-2.5 m: DRR roughly -2 to +6 dB.
- Far/off-axis speaker at 3-5 m: DRR roughly -8 to 0 dB.

These are engineering ranges. Actual DRR depends strongly on room size, absorption, mic orientation, and source directivity.

### Pyroomacoustics Example

```python
import numpy as np
import soundfile as sf
import pyroomacoustics as pra

sr = 16000
speech, _ = sf.read("dry_turn.wav")
if speech.ndim > 1:
    speech = speech.mean(axis=1)

# Example living room: 4.5 m x 5.0 m x 2.7 m
room_dim = [4.5, 5.0, 2.7]
rt60 = 0.45
absorption, max_order = pra.inverse_sabine(rt60, room_dim)

room = pra.ShoeBox(
    room_dim,
    fs=sr,
    materials=pra.Material(absorption),
    max_order=max_order,
    air_absorption=True,
)

# Phone on table near center, height 0.75 m
mic = np.array([[2.2], [2.4], [0.75]])
room.add_microphone_array(pra.MicrophoneArray(mic, fs=sr))

# Speaker standing/sitting, height around mouth level
source = [1.1, 2.0, 1.35]
room.add_source(source, signal=speech)

room.simulate()
out = room.mic_array.signals[0]
out = out / (np.max(np.abs(out)) + 1e-9) * 0.9
sf.write("room_turn.wav", out, sr, subtype="PCM_16")
```

For a multi-speaker dialogue, place each speaker at a different source location and keep the mic fixed. Do not convolve every turn with a totally different room unless the scene changes.

### Device and Channel Simulation

#### Phone microphone simulation

A plausible phone/table chain:

1. high-pass filter at 60-120 Hz;
2. mild low-pass or high-shelf attenuation above 6.5-7.5 kHz for 16 kHz output;
3. presence shaping around 2-5 kHz;
4. random narrow notches/peaks from table reflections;
5. dynamic range compression / AGC;
6. optional mild saturation;
7. optional codec encode-decode.

Do **not** use a single 300-3400 Hz bandpass for all phone audio unless the task specifically targets narrowband telephony. A modern phone voice memo or messaging-app recording can be wideband.

#### Suggested EQ distributions

For 16 kHz phone-like wideband:

- HPF: 70-100 Hz, 2nd-4th order.
- LPF: 7.2-7.8 kHz, gentle; or no LPF if already 16 kHz anti-aliased.
- Low-mid cut: -1 to -4 dB around 180-350 Hz if muddy.
- Presence boost: +0 to +3 dB around 2.5-4.5 kHz.
- Air/high shelf: -0 to -5 dB above 6 kHz depending on off-axis/distance.
- Random comb/notch: 1-3 notches, Q 3-12, depth -1 to -5 dB between 500 Hz and 5 kHz.

For narrowband telephony:

- Bandpass: 300-3400 Hz.
- Codec: GSM-FR, AMR-NB, or low-bitrate Opus.
- Resample back to 16 kHz only after codec/bandpass if required by dataset format.

#### SoX example

```bash
# Wideband phone-like coloration, not narrowband telephony
sox dry.wav phone_wide.wav \
  highpass 80 \
  equalizer 280 1.0q -2.5 \
  equalizer 3400 0.8q 1.8 \
  lowpass 7600 \
  compand 0.005,0.08 -55,-55,-22,-18,-8,-6,0,-3 0 -90 0.1 \
  gain -n -3
```

### Background Noise and Ambience

#### Useful datasets

- MUSAN: music, speech, and noise corpus; useful for augmentation. https://www.openslr.org/17/
- DEMAND: multichannel environmental noise database with domestic, office, and outdoor environments. https://zenodo.org/record/1227121
- FSD50K: Freesound-derived environmental sound clips with many classes. https://zenodo.org/record/4060432
- AudioSet: large ontology/dataset from YouTube; licensing/access differs by clip. https://research.google.com/audioset/
- WHAM!/WHAMR!: speech/noise/reverb resources for separation research. https://wham.whisper.ai/

#### SNR targets

| Scenario | SNR range | Noise examples |
|---|---:|---|
| Very quiet room / close phone | 25-40 dB | faint HVAC, low room tone |
| Normal apartment phone-on-table | 15-30 dB | refrigerator, street rumble, distant voices |
| Office | 10-25 dB | keyboard, HVAC, chairs, distant talk |
| Kitchen/living room activity | 8-20 dB | dishes, fan, footsteps, TV bleed |
| Phone in pocket/bag | 0-12 dB | cloth rustle, muffling, body occlusion |
| Outdoor/vehicle | 0-18 dB | traffic, wind, engine |

For classifier training, sample SNR from a distribution rather than fixed values. Example:

- 50% of clips: 18-30 dB;
- 30% of clips: 10-18 dB;
- 10% of clips: 5-10 dB;
- 10% of clips: 30-40 dB or nearly clean.

#### Noise mixing code

```python
import numpy as np


def rms(x):
    return np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12)


def mix_at_snr(clean, noise, snr_db, rng=np.random.default_rng()):
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)

    start = rng.integers(0, len(noise) - len(clean) + 1)
    noise = noise[start:start + len(clean)]

    noise = noise - np.mean(noise)
    target_noise_rms = rms(clean) / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / rms(noise))

    mixed = clean + noise
    peak = np.max(np.abs(mixed)) + 1e-9
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.astype(np.float32)
```

### Reverberation Details

#### Early reflections

Early reflections occur within roughly the first 5-80 ms after the direct sound. They strongly affect perceived distance and room geometry. For phone-on-table audio, early reflections from the table and nearby walls are more important than a long reverb tail.

#### Late reverb

Late reverb contributes room size and surface impression. Too much late reverb reduces intelligibility and can make the data unrealistic for domestic/office phone recordings. Keep RT60 mostly below 0.8 s for apartment/office scenes.

#### Recommended RIR strategy

- Use real RIRs for 50-70% of final clips.
- Use simulated RIRs for coverage and controlled variation.
- Keep one RIR/device/noise scene consistent across an entire 2-5 minute clip.
- Vary speaker distance by speaker, not randomly per sentence.
- Apply room before device/mic EQ, unless the RIR already includes the microphone/device coloration.

### Codec Simulation

#### Codecs

- **GSM-FR:** classic narrowband telephone sound.
- **AMR-NB:** 8 kHz narrowband mobile speech codec, bitrates up to 12.2 kbps.
- **AMR-WB:** wideband, 50-7000 Hz, bitrates 6.6-23.85 kbps.
- **Opus:** flexible from narrowband low-bitrate speech to fullband; common in WebRTC and messaging apps.

#### FFmpeg examples

```bash
# Narrowband AMR-NB simulation; requires FFmpeg built with libopencore-amrnb
ffmpeg -y -i in.wav -ar 8000 -ac 1 -c:a libopencore_amrnb -b:a 12.2k tmp.amr
ffmpeg -y -i tmp.amr -ar 16000 -ac 1 out_amrnb_16k.wav

# Opus wideband-ish messaging simulation
ffmpeg -y -i in.wav -ar 16000 -ac 1 -c:a libopus -b:a 24k tmp.opus
ffmpeg -y -i tmp.opus -ar 16000 -ac 1 out_opus_16k.wav

# More degraded Opus
ffmpeg -y -i in.wav -ar 16000 -ac 1 -c:a libopus -b:a 12k -application voip tmp.opus
ffmpeg -y -i tmp.opus -ar 16000 -ac 1 out_opus_low_16k.wav
```

Do not apply codec simulation to every clip. Include a label/metadata field for codec condition so you can audit whether the classifier is learning codec artifacts.

### Low-Pass and High-Pass Filtering at 16 kHz

At 16 kHz sample rate, Nyquist is 8 kHz. The speech band above 7.5 kHz is small but can contain fricative brightness and noise. A single well-designed low-pass at 7.5 kHz should not cause severe muffling, but it may if:

- the filter rolloff starts much lower;
- the filter is applied repeatedly;
- the filter has passband ripple or phase artifacts;
- it is combined with denoising;
- codec simulation or EQ also removes 3-7 kHz detail.

Recommended filters:

| Purpose | Filter |
|---|---|
| Remove rumble | HPF 50-80 Hz for normal speech; 80-120 Hz for phone-like audio |
| Preserve clarity | avoid cutting 2-5 kHz; this is critical for intelligibility |
| Anti-alias at 16 kHz | LPF 7.6-7.9 kHz, gentle, if needed |
| Phone wideband | HPF 80 Hz, mild LPF 7.2-7.6 kHz, device EQ |
| Narrowband telephony | 300-3400 Hz bandpass, use only for narrowband condition |

### Denoising on Clean TTS

Wiener denoising estimates noise and suppresses spectral components. On clean TTS, it can:

- attenuate fricatives and breath components;
- create musical noise;
- smear transients;
- reduce high-frequency detail;
- make speech sound underwater/muffled;
- remove precisely the micro-noise that helps synthetic speech sound human.

Recommendation:

- Remove Wiener denoising from clean TTS.
- If noise must be cleaned, denoise the noise source before mixing or use a high-quality speech enhancement model only on deliberately noisy final audio.
- Prefer controlled noise mixing over denoise-then-re-noise.
- For clipping/clicks, use de-clicking and crossfades, not broadband denoising.

---

## Part 4: Pitch and Prosody Engineering

### Angry Speech vs. Calm Speech

Hot anger differs from calm speech in several dimensions:

| Feature | Calm | Hot anger | Cold anger |
|---|---|---|---|
| Mean F0 | baseline or slightly low | often higher, but speaker-dependent | stable or slightly low/high |
| F0 range | moderate/narrow | wide | moderate with sharp accents |
| Rate | moderate/slow | faster articulation | controlled, clipped |
| Pauses | longer, softer boundaries | shorter response gaps, fewer long pauses | strategic pauses, tense silence |
| Intensity | moderate-low | high | medium-high but restrained |
| Spectral tilt | normal/steeper | flatter, brighter | tense, sometimes creaky |
| Articulation | relaxed | precise/forceful, consonant bursts | precise, clipped |
| Voice quality | modal | pressed/tense, sometimes harsh | tense/creaky/low-energy threat |

For your reported issue, implement male anger as:

- median F0: baseline to +1 semitone for most angry turns;
- F0 90th percentile: +2 to +5 semitones for accented words;
- F0 SD: +30-80%;
- articulation rate: +10-25%;
- pause duration: reduce by 20-50%;
- RMS: +3-8 dB before AGC/compression;
- high-frequency ratio: +10-30% via spectral tilt/presence boost, not harsh clipping;
- phrase-final falls sharper or level/low for cold anger.

### Male vs. Female Pitch Ranges for Hebrew Speakers

I did not find a robust, widely cited open table of Hebrew adult conversational F0 norms comparable to general voice-science norms. Use the following as engineering defaults and calibrate against Hebrew corpora.

| Speaker | Neutral median F0 target | Guardrail | Notes |
|---|---:|---:|---|
| Male Hebrew voice | 105-135 Hz | reject sustained median >170 Hz unless speaker identity supports it | Emotional peaks can exceed 200 Hz briefly. |
| Female Hebrew voice | 175-230 Hz | reject sustained median <150 Hz unless voice identity supports it | Low female pitch can be natural for some speakers, but TTS often sounds artificially pitched down. |
| Male distress/fear | 125-180 Hz median possible | avoid monotonic upward drift across clip | Use broken contours and breath rather than global pitch. |
| Female calm/low intensity | 165-215 Hz | avoid pushing below ~150 Hz for long spans | Use softer intensity/rate instead of heavy pitch lowering. |

Pipeline requirement: estimate each synthetic speaker’s baseline from neutral renders and store it. All emotional F0 targets should be relative to that baseline.

### How Pitch Should Change Across an Escalating Argument

Escalation should not mean every turn has higher mean pitch than the previous turn. Real arguments often alternate between spikes, controlled low anger, interruptions, and fatigue.

Example escalation plan for a male speaker with neutral median F0 = 120 Hz:

| Stage | Median F0 | F0 range | Rate | Intensity | Example delivery |
|---|---:|---:|---:|---:|---|
| 1: calm disagreement | 118-124 Hz | baseline | baseline | baseline | controlled, explanatory |
| 2: irritation | 120-130 Hz | +20% | +5-10% | +2-4 dB | shorter pauses |
| 3: accusation | 122-135 Hz | +50% | +10-20% | +5-8 dB | accented peaks, sharper consonants |
| 4: controlled anger | 115-128 Hz | +30% | +5-15% | +4-7 dB | lower/tense, clipped timing |
| 5: shout/spike | 130-160 Hz | +80% | +15-30% | +8-15 dB | short burst, possible saturation |
| 6: aftermath | 110-125 Hz | -10 to +20% | -10% | -3 dB | breathy/fatigued, longer pauses |

Use per-turn emotional states. Avoid a linear upward F0 drift across a 2-5 minute clip. If a TTS engine drifts upward during long synthesis, split turns and reset.

### Pitch Level vs. Pitch Variation

This distinction is critical:

- **Pitch level:** median or mean F0.
- **Pitch variation:** range, SD, contour shape, accent peaks.

For anger, especially male anger, increase variation and local peaks more than global median. For fear, both median and range may rise. For distress, median may fall, but range can include sudden upward breaks.

Useful metrics:

```text
median_f0_hz
f0_p10_hz
f0_p90_hz
f0_range_semitones = 12 * log2(p90 / p10)
f0_sd_semitones
turn_to_turn_median_f0_delta_semitones
slope_of_median_f0_over_clip
```

### Preventing Pitch Drift in Multi-Turn TTS Rendering

Pitch drift is often caused by long-context synthesis, style conditioning, punctuation patterns, or SSML accumulation.

Mitigations:

1. Render each turn independently.
2. Keep one voice ID per speaker.
3. Reset SSML state every request.
4. Do not synthesize a full 2-5 minute conversation as one TTS request.
5. Avoid nested prosody over long spans.
6. Analyze F0 after every turn.
7. Reject/resynthesize turns outside speaker guardrails.
8. Use text-level emotional cues before pitch tags.
9. For turn continuity, join audio in post, not through one long TTS context.
10. Store per-speaker F0 baselines and enforce drift thresholds.

Suggested rejection threshold:

- Per-turn median F0 must be within speaker profile range.
- For same speaker and same nominal emotion, median F0 drift across clip should stay within ±2 semitones unless intentionally scripted.
- A sustained monotonic F0 slope over many turns is suspicious; reject if median F0 increases for 4+ consecutive turns without scripted escalation.

### Rate Manipulation for Hebrew

Open, robust Hebrew-specific speaking-rate norms are sparse. Use these as initial engineering targets and refine using real Hebrew corpora.

| State | Articulation rate target | Overall rate target | Notes |
|---|---:|---:|---|
| Calm | 3.8-5.2 syll/s | 3.0-4.5 syll/s including pauses | Do not make every sentence slow. |
| Neutral dialogue | 4.5-6.2 syll/s | 3.5-5.3 syll/s | Good default. |
| Argument / irritation | 5.3-7.0 syll/s | 4.3-6.2 syll/s | Shorter pauses and faster response gaps. |
| Hot anger | 5.8-7.5 syll/s | 4.8-6.8 syll/s | Short bursts can exceed this. |
| Fear/anxiety | 5.0-7.0 syll/s | variable | Include hesitations/breaths. |
| Distress | 3.2-5.2 syll/s | 2.5-4.5 syll/s | Pauses and broken phrasing matter. |

Use forced alignment or syllable approximations to measure rate. Hebrew syllabification can be approximated from vowels/diacritics if present, but for unvocalized Hebrew, a robust syllable counter is nontrivial. For gating, character/phone duration, ASR word rate, and TTS text token duration may be more practical.

### Loudness and RMS: Real Shouting vs. TTS “Loud”

Real shouting changes:

- SPL/intensity;
- spectral tilt, with more high-frequency energy;
- glottal closure and pressed phonation;
- articulation force;
- breath support;
- distance behavior through AGC/compression;
- clipping/limiting risk in phone recordings.

TTS `volume="loud"` mainly affects output level or model loudness conditioning. It does not reliably create vocal effort. To simulate shouting:

1. Use text and punctuation: shorter clauses, direct address, interruptions.
2. Use mild TTS rate and pitch changes.
3. Increase local level pre-device.
4. Apply dynamic compression/limiting as a phone would.
5. Add spectral presence/brightness.
6. Optionally add slight saturation on peaks.
7. Keep room/device response consistent.

Loudness targets for final clips:

- Integrated loudness: approximately -26 to -18 LUFS depending scenario.
- Peak: below -1 dBFS, preferably -3 dBFS before final PCM export.
- Turn RMS: vary by emotion and distance; do not normalize every turn to identical RMS.
- Crest factor: avoid over-compressed speech below ~6 dB crest factor unless intentionally phone/AGC-heavy.

### Adding Micro-Prosody in Post-Processing

Can jitter, shimmer, creak, or breathiness be added after TTS? Yes, but with caveats.

#### What works reasonably

- subtle local F0 perturbation using WORLD/PSOLA on voiced frames;
- phrase-final creak by lowering/irregularizing F0 on final voiced segments;
- breath sounds as separate events;
- mild amplitude modulation for trembling/fear;
- spectral tilt/EQ/saturation for vocal effort;
- adding room/device/noise realism.

#### What is risky

- random waveform jitter;
- global vibrato/tremolo;
- heavy pitch shifting after TTS;
- breath noise over all speech;
- creak applied to all low-pitched segments;
- shimmer that tracks consonants and silence.

#### WORLD-based F0 perturbation example

```python
import numpy as np
import pyworld as pw
import soundfile as sf


def add_subtle_f0_microvariation(x, sr, amount_semitones=0.08, corr_frames=5, seed=0):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)

    f0, t = pw.harvest(x, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    voiced = f0 > 0
    noise = rng.normal(0, amount_semitones, size=len(f0))

    # Smooth perturbation so it is not frame-random.
    kernel = np.ones(corr_frames) / corr_frames
    noise = np.convolve(noise, kernel, mode="same")

    f0_new = f0.copy()
    f0_new[voiced] *= 2 ** (noise[voiced] / 12.0)

    y = pw.synthesize(f0_new, sp, ap, sr)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.98
    return y.astype(np.float32)

x, sr = sf.read("tts_turn.wav")
y = add_subtle_f0_microvariation(x, sr, amount_semitones=0.06)
sf.write("tts_turn_micro.wav", y, sr, subtype="PCM_16")
```

Use this only after A/B testing. WORLD re-synthesis can degrade high-quality TTS, especially with noisy consonants. For many pipelines, better gains come from room/device/noise and prosody gating rather than artificial jitter.

---

## Part 5: Eliminating TTS Artifacts

### Common Azure Neural TTS Artifacts and Likely Causes

| Artifact | Likely cause | Fix |
|---|---|---|
| 16-second sustained vowel | synthesis failure, malformed text, unsupported SSML, long context, repeated punctuation, endpoint glitch | Reject automatically; resynthesize with shorter text and simpler SSML. |
| Clicks at sentence boundaries | hard concatenation, non-zero crossing cuts, codec priming, DC offset | Trim at zero crossings, use 10-30 ms equal-power crossfades. |
| Voice identity shifts | switching styles/roles/voices, styledegree too high, using unsupported style, long request drift | Fixed voice per speaker; avoid style for he-IL; synthesize turns independently. |
| Robotic pacing | global slow rate, uniform pauses, written text, TTS default cadence | Rewrite for dialogue; vary pauses; use local rate changes; insert overlaps. |
| Muffled sound | denoising, excessive EQ/low-pass, codec overuse, high-frequency attenuation | Remove Wiener denoise; preserve 2-7 kHz; audit spectral ratios. |
| Glitches at punctuation | unusual punctuation, nested SSML, long pauses | Normalize punctuation; keep breaks 80-450 ms; insert long pauses in post. |
| Pronunciation errors | Hebrew unvocalized ambiguity, names, acronyms, numbers | Use custom lexicon/phoneme tags if supported; pre-expand numbers/acronyms. |

### SSML Patterns That Can Cause Failures

Risky patterns:

- very long SSML documents;
- many nested `<prosody>` tags;
- unsupported `mstts:express-as` for the selected voice;
- `styledegree` near 2.0;
- word-level style tags;
- extreme `rate`, `pitch`, or `volume` values;
- multi-second `<break>` tags;
- repeated punctuation such as `!!!!!!!!!` or `.....`;
- weird Unicode control characters;
- mixed language text without explicit language handling;
- raw abbreviations/acronyms in Hebrew;
- combining SSML `audio` inserts with long text near service limits.

### Safe Parameter Ranges for Azure SSML

Recommended production ranges for Hebrew dialogue:

```text
rate:
  neutral: -3% to +5%
  calm: -8% to +2%
  anger: +4% to +12%
  fear: +2% to +10%, plus pauses/breaths
  distress: -12% to +2%

pitch:
  male neutral/calm: -3% to +2%
  male anger: -2% to +5%, use contour/range more than mean
  female neutral/calm: -2% to +3%
  female distress: -5% to +2%, avoid sustained very low median
  fear: +4% to +10% for short spans, not entire clip

breaks:
  comma-like: 80-180 ms
  clause boundary: 180-350 ms
  turn hesitation: 250-600 ms
  long dramatic pause: insert silence in post, not TTS, if >700 ms

volume:
  use medium/loud or modest numeric values;
  avoid making every emotional turn x-loud;
  let final device AGC/compression shape loudness.

styledegree, only if the voice/style is documented:
  0.3-0.8 default
  0.8-1.1 strong
  >1.2 only after manual validation
```

### Detecting and Filtering Broken TTS Output

Add automatic gates at the **turn level** and **final clip level**.

#### Turn-level gates

- Duration ratio: measured duration vs expected duration from text length.
- Silence ratio: too much silence or no silence.
- Max sustained voiced segment: reject if one voiced vowel-like segment lasts >2.5-3.0 s, unless scripted singing/crying.
- F0 bounds: reject if median F0 outside speaker profile.
- F0 drift: reject if same speaker drifts too far across turns.
- Spectral anomaly: reject if high-frequency ratio too low or sudden narrowband condition not intended.
- Clicks: detect large sample-to-sample jumps.
- Clipping: reject if >0.1% samples above -0.1 dBFS.
- ASR alignment: use Hebrew ASR to check that output matches script.
- Speaker embedding: compare to speaker baseline embedding.

#### Example quality-gate code

```python
import numpy as np
import librosa
import soundfile as sf


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def analyze_turn(path, speaker_profile=None):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    duration = len(y) / sr
    peak = np.max(np.abs(y)) + 1e-12
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)

    # Click proxy: large first differences relative to RMS.
    dy = np.diff(y)
    click_score = np.percentile(np.abs(dy), 99.95) / (rms + 1e-9)

    # Spectral high-frequency ratio for 16 kHz audio.
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160, win_length=400)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    total = S[(freqs >= 80) & (freqs <= 7800)].sum() + 1e-12
    high = S[(freqs >= 4000) & (freqs <= 7800)].sum()
    high_ratio_db = db(high / total)

    # F0 using pyin. Tune fmin/fmax per speaker if known.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C6"),   # ~1047 Hz, broad guard
        sr=sr,
        frame_length=1024,
        hop_length=160,
    )
    voiced_f0 = f0[np.isfinite(f0)]
    median_f0 = float(np.median(voiced_f0)) if len(voiced_f0) else np.nan
    p10 = float(np.percentile(voiced_f0, 10)) if len(voiced_f0) else np.nan
    p90 = float(np.percentile(voiced_f0, 90)) if len(voiced_f0) else np.nan
    f0_range_st = float(12 * np.log2(p90 / p10)) if p10 > 0 else np.nan

    # Sustained voiced run proxy.
    voiced = np.isfinite(f0)
    max_run = 0
    cur = 0
    for v in voiced:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    max_voiced_run_s = max_run * 160 / sr

    return {
        "duration_s": duration,
        "peak_dbfs": db(peak),
        "rms_dbfs": db(rms),
        "click_score": click_score,
        "high_ratio_db": high_ratio_db,
        "median_f0_hz": median_f0,
        "f0_range_st": f0_range_st,
        "max_voiced_run_s": max_voiced_run_s,
    }


def reject_reason(metrics):
    reasons = []
    if metrics["duration_s"] < 0.5:
        reasons.append("too_short")
    if metrics["peak_dbfs"] > -0.1:
        reasons.append("clipping_risk")
    if metrics["click_score"] > 80:
        reasons.append("possible_clicks")
    if metrics["max_voiced_run_s"] > 2.8:
        reasons.append("sustained_voiced_segment")
    if metrics["high_ratio_db"] < -35:
        reasons.append("muffled_or_narrowband")
    return reasons
```

Thresholds must be calibrated on your own clean and known-bad TTS outputs.

### Crossfading and Segment Joining

Clicks occur when waveforms are joined at discontinuities. Use trim + fade + crossfade.

#### Equal-power crossfade

```python
import numpy as np


def equal_power_crossfade(a, b, sr, ms=20):
    n = int(sr * ms / 1000)
    n = min(n, len(a), len(b))
    if n <= 0:
        return np.concatenate([a, b])

    t = np.linspace(0, np.pi / 2, n)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    mid = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], mid, b[n:]])
```

#### Recommended join workflow

1. Remove leading/trailing DC offset.
2. Trim only excessive silence; preserve natural breath/room tone if present.
3. Apply 5-10 ms fade-in/fade-out to isolated turns.
4. For adjacent speech regions, use 10-30 ms equal-power crossfade.
5. Add ambience under the entire clip after joining, so silence has room tone.
6. If using room convolution, convolve before final joining or maintain RIR tails carefully.

### Voice Consistency Across Turns

Speaker identity consistency is a dataset requirement. Otherwise, the classifier may learn identity drift as an artifact.

Recommendations:

- Assign each synthetic speaker a fixed TTS engine, voice, baseline pitch range, baseline rate, and device position.
- Avoid switching `express-as` styles for Hebrew voices unless verified stable.
- Use emotion via script, rate, contour, local intensity, and post-processing.
- Generate neutral anchor lines for each speaker and compute speaker embeddings using ECAPA-TDNN or x-vector.
- Reject a turn if its speaker embedding is too far from the speaker’s anchor distribution.

Tools:

- SpeechBrain ECAPA-TDNN speaker embeddings: https://speechbrain.github.io/
- NVIDIA NeMo speaker models: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/intro.html
- pyannote.audio embeddings/diarization: https://github.com/pyannote/pyannote-audio

Example consistency idea:

```python
# Pseudocode
anchor_emb = mean_embedding(neutral_anchor_turns)
turn_emb = speaker_embedding(candidate_turn)
cosine = np.dot(anchor_emb, turn_emb) / (np.linalg.norm(anchor_emb) * np.linalg.norm(turn_emb))

if cosine < 0.72:   # calibrate per embedding model and TTS engine
    reject("speaker_identity_shift")
```

---

## Part 6: Existing Projects and Tools

### Open-Source Projects for Synthetic Speech Data and Augmentation

#### Lhotse

Lhotse is a speech/audio data preparation toolkit used with ASR/TTS pipelines. It provides manifests, cuts, supervision segments, and augmentation workflows.

- URL: https://github.com/lhotse-speech/lhotse
- Use case: build reproducible manifests for generated Hebrew dialogue, including speaker IDs, room/device/noise metadata, emotion labels, and quality metrics.

#### SpeechBrain

SpeechBrain provides speech models, including speaker recognition, enhancement, ASR, and recipes.

- URL: https://speechbrain.github.io/
- Use case: speaker-embedding consistency gates and augmentation experiments.

#### NVIDIA NeMo

NeMo provides ASR, speaker recognition, TTS, and speech data tooling.

- URL: https://github.com/NVIDIA/NeMo
- Use case: speaker embeddings, ASR checks, possible TTS experimentation.

#### ESPnet

ESPnet includes ASR/TTS recipes and augmentation utilities.

- URL: https://github.com/espnet/espnet
- Use case: training/evaluating speech models and checking open TTS pipelines.

#### Scaper

Scaper is a soundscape synthesis library that can mix events and backgrounds with labels.

- URL: https://github.com/justinsalamon/scaper
- Use case: domestic/office sound events under dialogue.

#### pyroomacoustics and gpuRIR

Covered above. Use them for simulated RIRs and microphone geometry.

### Relevant Datasets

#### RAVDESS

The Ryerson Audio-Visual Database of Emotional Speech and Song contains 7,356 files from 24 professional actors, with emotions including calm, happy, sad, angry, fearful, surprise, disgust, and neutral. Audio-only files are 16-bit, 48 kHz WAV.

- Paper: Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
- Dataset: https://zenodo.org/records/1188976

Use: emotional acoustic reference, not Hebrew. Good for studying acted emotion profiles and quality gates.

#### IEMOCAP

IEMOCAP contains approximately 12 hours of audiovisual data from dyadic sessions with scripted and improvised interactions designed to elicit emotions such as happiness, anger, sadness, frustration, and neutral state.

- Paper: Busso, C., et al. (2008). *IEMOCAP: Interactive emotional dyadic motion capture database*. https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf
- Dataset page: https://sail.usc.edu/iemocap/

Use: dialogue emotion dynamics, turn-taking, spontaneous vs acted emotion. Not Hebrew.

#### ESD

The Emotional Speech Dataset contains 350 parallel utterances spoken by 10 native English and 10 native Mandarin speakers in 5 emotion categories: neutral, happy, angry, sad, and surprise. It contains more than 29 hours of controlled recordings.

- Paper: Zhou, K., Sisman, B., Liu, R., & Li, H. (2021/2022). *Emotional Voice Conversion: Theory, Databases and ESD*. https://arxiv.org/abs/2105.14762
- Dataset page: https://hltsingapore.github.io/ESD/
- GitHub: https://github.com/HLTSingapore/Emotional-Speech-Data

Use: parallel emotional speech for studying acoustic transformations.

#### LibriTTS and LibriTTS-R

LibriTTS is a 585-hour multi-speaker English TTS corpus at 24 kHz from 2,456 speakers, derived from LibriSpeech/LibriVox. LibriTTS-R is a sound-quality-improved version.

- LibriTTS OpenSLR: https://www.openslr.org/60/
- Paper: Zen, H., et al. (2019). *LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech*. https://arxiv.org/abs/1904.02882
- LibriTTS-R: https://www.openslr.org/141/

Use: TTS dataset design and quality practices; not emotional dialogue.

#### VCTK

The CSTR VCTK Corpus contains speech from 110 English speakers with various accents, each reading about 400 sentences.

- Dataset: https://datashare.ed.ac.uk/handle/10283/3443

Use: multi-speaker voice quality and speaker-embedding experiments.

#### Mozilla Common Voice

Common Voice is a public, community-contributed multilingual speech dataset. The Mozilla site describes 130+ languages; the current release ecosystem includes many more languages and ongoing scripted/spontaneous speech releases. Hebrew is available, though the amount and quality should be checked in the latest dataset page.

- Main site: https://commonvoice.mozilla.org/
- Datasets: https://commonvoice.mozilla.org/datasets
- Release metadata: https://github.com/common-voice/cv-dataset

Use: Hebrew speaker variation, accent/noise references, ASR training/evaluation. Common Voice is not controlled emotional speech.

#### Hebrew-specific resources

- SASPEECH: 30-hour single-speaker Hebrew corpus on OpenSLR. https://www.openslr.org/134/
- HebTTS: diacritics-free Hebrew TTS research. https://github.com/slp-rl/HebTTS
- Mamre-TTS: Hebrew TTS project. https://github.com/mamre-tts/mamre-tts
- Israwave: Hebrew TTS project. https://github.com/thewh1teagle/israwave
- Phonikud: Hebrew G2P / pronunciation tooling. https://github.com/thewh1teagle/phonikud
- eSpeak NG Hebrew support may be useful for pronunciation prototyping, not natural TTS. https://github.com/espeak-ng/espeak-ng

### Audio Augmentation Libraries

#### audiomentations

- URL: https://github.com/iver56/audiomentations
- CPU-based NumPy augmentation library.
- Good for offline dataset generation.
- Supports transforms such as AddBackgroundNoise, ApplyImpulseResponse, BandPassFilter, Gain, ClippingDistortion, AddShortNoises, TimeStretch, PitchShift.

#### torch-audiomentations

- URL: https://github.com/asteroid-team/torch-audiomentations
- PyTorch/GPU-friendly augmentation.
- Good for training-time augmentation.
- Useful if augmentation should happen inside a model pipeline.

#### WavAugment

- URL: https://github.com/facebookresearch/WavAugment
- Applies effects to PyTorch tensors.
- Effects include pitch, reverb, temporal masking, band rejection, clipping.
- Useful for speech tasks, but validate artifacts carefully.

#### SoX effects

- URL: https://sox.sourceforge.net/sox.html
- Very reliable for deterministic command-line audio effects.
- Useful for EQ, filtering, companding, reverb, trimming, fades, and splicing.

#### FFmpeg

- URL: https://ffmpeg.org/
- Essential for codec simulation and format conversion.

#### librosa / soundfile / pyloudnorm / Praat-Parselmouth

- librosa: https://librosa.org/
- python-soundfile: https://python-soundfile.readthedocs.io/
- pyloudnorm: https://github.com/csteinmetz1/pyloudnorm
- Parselmouth: https://github.com/YannickJadoul/Parselmouth

Use for analysis, loudness, F0, spectral features, and quality gates.

### Room Impulse Response Databases

#### MIT IR Survey

Real-world impulse responses collected from everyday spaces.

- URL: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html

#### OpenAIR

Open Acoustic Impulse Response database with many spaces.

- URL: https://www.openair.hosted.york.ac.uk/

#### EchoThief

A collection of impulse responses from many unique spaces.

- URL: http://www.echothief.com/

Check license terms carefully. EchoThief has permissive convolution/derivative-use language but may require contacting the creator for other uses.

#### BUT ReverbDB

Database with real RIRs, environmental noises/silences, retransmitted speech, and metadata.

- URL: https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database
- Paper: https://arxiv.org/abs/1609.03002

#### OpenSLR RIRS_NOISES

Room impulse response and noise database with simulated and real RIRs, isotropic noises, and point-source noises at 16 kHz/16-bit.

- URL: https://www.openslr.org/28/

### Speech Quality and Naturalness Assessment Tools

#### PESQ

PESQ is an ITU full-reference objective speech quality metric for narrowband/wideband telephony. It requires a clean reference. It is useful for codec/noise degradation but not ideal for TTS naturalness.

- ITU-T P.862: https://www.itu.int/rec/T-REC-P.862

#### POLQA

POLQA is a newer full-reference ITU metric, standardized as P.863, commonly used for modern telecommunication quality. It is licensed/commercial in many workflows.

- ITU-T P.863: https://www.itu.int/rec/T-REC-P.863

#### DNSMOS

DNSMOS is a no-reference speech quality metric developed for noise suppression evaluation. It estimates SIG, BAK, and OVRL quality dimensions and is useful for detecting noise/reverb/codec degradation, but it is not a complete TTS naturalness score.

- DNS Challenge / DNSMOS: https://github.com/microsoft/DNS-Challenge
- DNSMOS paper/resources: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS

#### NISQA

NISQA is a no-reference speech quality model that predicts overall quality and dimensions such as noisiness, coloration, discontinuity, and loudness.

- URL: https://github.com/gabrielmittag/NISQA
- Paper/corpus details: https://arxiv.org/abs/2104.09494

#### UTMOS / VoiceMOS

UTMOS is a neural MOS predictor for synthetic speech naturalness and was prominent in VoiceMOS challenges. It is closer to TTS naturalness evaluation than PESQ/DNSMOS, but it should not replace human listening.

- UTMOS v2 implementation: https://github.com/sarulab-speech/UTMOSv2
- VoiceMOS challenge context: https://voicemos-challenge.github.io/

#### Recommended metric stack

Use multiple metrics because no single metric catches all failure modes:

| Metric/tool | Use | Gate idea |
|---|---|---|
| ASR WER/CER against script | intelligibility and severe artifacts | reject if CER too high for clean/wideband condition |
| F0 stats | pitch guardrails/drift | reject out-of-profile turns |
| High-frequency ratio / spectral centroid | muffling/narrowband detection | reject unintended low HF ratio |
| Click detector | boundary artifacts | reject or repair high click score |
| Max voiced run | sustained vowel failure | reject >2.5-3 s |
| DNSMOS/NISQA | noise/coloration/discontinuity | reject severe degradation |
| UTMOS/VoiceMOS | TTS naturalness proxy | use for ranking candidates |
| Speaker embedding cosine | identity consistency | reject style-induced identity shifts |
| Human spot checks | final validation | sample each batch and condition |

---

## Part 7: Recommended Architecture Changes

### Immediate Changes to Remove or Modify

#### Remove Wiener denoising from clean TTS

Current issue: clean TTS is already noise-free. Wiener denoising can remove fricatives, breath components, and high-frequency detail.

Action:

- Delete Wiener denoising from the default path.
- Keep denoising only for optional experiments on noisy final mixes.
- Do not denoise before adding noise/room.

Expected benefit: high. Effort: low.

#### Stop treating 7.5 kHz low-pass as a generic realism step

At 16 kHz, Nyquist is 8 kHz. A 7.5 kHz LPF should be only an anti-alias/safety filter, not a “phone realism” filter.

Action:

- Use no LPF or a gentle 7.6-7.9 kHz LPF for standard wideband outputs.
- For phone-wideband coloration, use EQ and device simulation instead of only low-pass.
- For narrowband telephony condition, explicitly bandpass 300-3400 Hz and label it as narrowband.

Expected benefit: medium-high. Effort: low.

#### Do not synthesize 2-5 minute clips as one TTS request

Action:

- Synthesize per speaker turn or per sentence group.
- Reset SSML for every request.
- Join and spatialize in post.

Expected benefit: high for pitch drift/artifact control. Effort: medium.

#### Avoid unsupported Azure `express-as` for Hebrew

Action:

- Treat Avri/Hila as neutral/prosody-controlled voices unless Microsoft’s current docs for your region list style support.
- If you use styles, test style support automatically by rendering a probe and measuring identity/F0/timing.
- Do not switch styles across turns for the same speaker without speaker-embedding gates.

Expected benefit: high for identity consistency. Effort: low-medium.

### Recommended TTS Generation Architecture

#### Speaker profile

Create a fixed profile per synthetic speaker:

```yaml
speaker_id: spk_001
engine: azure
voice: he-IL-AvriNeural
sex_label: male
neutral_f0_median_hz: 122
neutral_rate_syll_per_s: 5.2
pitch_guardrail_hz: [85, 180]
emotion_policy:
  calm:
    rate_pct: [-6, 2]
    pitch_pct: [-3, 2]
  anger_hot:
    rate_pct: [4, 12]
    pitch_pct: [-1, 5]
    f0_range_multiplier: [1.3, 1.8]
  anger_cold:
    rate_pct: [-2, 8]
    pitch_pct: [-4, 1]
  fear:
    rate_pct: [2, 10]
    pitch_pct: [4, 10]
scene_position:
  azimuth_deg: -25
  distance_m: 1.4
```

#### Candidate generation

For every turn:

1. Generate 2-5 candidates with mild random prosody around the target.
2. Analyze duration, F0, spectral ratio, clicks, sustained voiced runs.
3. Run ASR alignment if available.
4. Score naturalness with UTMOS/NISQA/DNSMOS as appropriate.
5. Select best candidate or resynthesize.

Example scoring:

```python
score = 0.0
score += 1.0 * normalized_utmos
score += 0.5 * normalized_nisqa
score -= 2.0 * sustained_vowel_penalty
score -= 1.5 * f0_out_of_profile_penalty
score -= 1.0 * click_penalty
score -= 1.0 * asr_cer_penalty
score -= 1.0 * speaker_embedding_penalty
```

### Recommended SSML Changes

#### Use mild prosody and phrase-level contour

For Hebrew anger, avoid global high pitch. Use rate, shorter breaks, and local contour.

```xml
<speak version="1.0" xml:lang="he-IL"
       xmlns="http://www.w3.org/2001/10/synthesis">
  <voice name="he-IL-AvriNeural">
    <prosody rate="+8%" pitch="-1%" volume="loud">
      לא, תקשיב לי רגע.
      <break time="120ms"/>
      זה בדיוק מה שאמרתי שלא יקרה.
    </prosody>
  </voice>
</speak>
```

#### Use pauses as dialogue actions

Instead of uniform SSML breaks, generate timeline pauses:

```yaml
turns:
  - speaker: A
    text: "לא, תקשיב לי רגע."
    tts_breaks: [120]
  - gap_after_previous_ms: 80
  - speaker: B
    text: "אני מקשיב."
  - overlap_with_previous_ms: 120
```

Insert long silences, overlaps, and interruptions in the dialogue timeline rather than inside TTS.

#### Add disfluencies carefully

Hebrew dialogue can include fillers and repairs:

```text
רגע, רגע— לא לזה התכוונתי.
אמ... אני לא בטוחה שזה נכון.
תקשיב, זה לא עובד ככה.
```

Do not overuse fillers. Use them as event labels: hesitation, interruption, repair, breath, sigh.

### Recommended Post-Processing Chain

#### Proposed chain

```text
TTS dry turn
  -> turn quality gate
  -> trim/fade/crossfade prep
  -> dialogue timeline assembly
  -> per-speaker room convolution / spatial mix
  -> device EQ + AGC/compression
  -> background ambience + sound events
  -> optional codec condition
  -> loudness normalization / peak limiting
  -> final 16 kHz mono PCM export
  -> final quality gates
```

#### Example audiomentations chain

```python
from audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse, Gain, ClippingDistortion

augment = Compose([
    ApplyImpulseResponse(
        ir_path="rir_database/",
        p=0.7,
        leave_length_unchanged=False,
    ),
    AddBackgroundNoise(
        sounds_path="noise_database/",
        min_snr_db=12,
        max_snr_db=30,
        p=0.9,
    ),
    Gain(min_gain_db=-3, max_gain_db=3, p=0.5),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=3, p=0.08),
])

y_aug = augment(samples=y, sample_rate=16000)
```

For scene consistency, do not call a fully random augmenter independently for every turn. Sample a scene profile once per clip and apply consistent settings.

#### Device AGC/compression example

```bash
sox room_mix.wav device_mix.wav \
  highpass 80 \
  equalizer 250 1.0q -2.0 \
  equalizer 3600 0.9q 1.5 \
  compand 0.003,0.08 -60,-60,-30,-24,-12,-9,0,-4 0 -90 0.1 \
  gain -n -3
```

### Quality Gates and Thresholds

Calibrate thresholds on real and synthetic validation sets. Initial defaults:

#### Turn-level thresholds

| Metric | Reject / flag threshold | Notes |
|---|---:|---|
| Max sustained voiced segment | >2.8 s reject | catches 16 s vowel failures |
| Peak level | >-0.1 dBFS reject | clipping risk |
| Click score | above calibrated p99 of good data | repair or reject |
| Male median F0 | outside 80-180 Hz for normal/anger; profile-specific | allow rare peaks, not sustained median |
| Female median F0 | outside 150-280 Hz for normal/anger; profile-specific | lower allowed only for specific voices |
| F0 range | <2 st for emotional turn flag; >18 st flag | depends on text length |
| Duration ratio | >1.5x or <0.65x expected flag | use text/phoneme duration model |
| High-frequency ratio | below calibrated floor | catches muffling/unintended narrowband |
| ASR CER | >0.15 clean; >0.25 noisy flag | Hebrew ASR dependent |
| Speaker embedding cosine | <0.70-0.80 flag | calibrate model-specific |

#### Clip-level thresholds

| Metric | Target |
|---|---:|
| Duration | 2-5 min |
| Speech activity ratio | 35-85% depending dialogue density |
| Integrated loudness | -26 to -18 LUFS, condition-specific |
| True peak | <= -1 dBTP if using loudness tooling; otherwise <= -3 dBFS peak before PCM export |
| Noise SNR | metadata target within sampled range |
| RT60 | metadata target 0.25-0.85 s for most rooms |
| Speaker drift | no unexplained monotonic median F0 drift >2 semitones |
| Artifact count | zero sustained-vowel failures; zero severe clicks |

### Priority Ordering: Biggest Quality Improvement for Least Effort

#### Priority 1: Remove damaging processing

- Remove Wiener denoising from clean TTS.
- Stop aggressive or repeated low-pass filtering.
- Preserve 2-7 kHz clarity.
- Add final high-pass only for rumble.

Expected impact: very high. Effort: low.

#### Priority 2: Turn-level synthesis and artifact gates

- Render one turn at a time.
- Add sustained-vowel detection.
- Add click detection.
- Add F0 guardrails.
- Add duration-ratio checks.

Expected impact: very high. Effort: medium.

#### Priority 3: Fix emotional prosody policy

- Stop mapping anger to global pitch rise.
- Use per-speaker baselines.
- Separate mean F0 from F0 range.
- Use rate/intensity/spectral tilt/articulation for anger.
- Keep female low-pitch guardrails.

Expected impact: high. Effort: medium.

#### Priority 4: Add realistic scene simulation

- Use RIRs with RT60 0.25-0.85 s.
- Add consistent room/device/noise profiles per clip.
- Use phone-table EQ/AGC.
- Mix realistic ambience at 10-30 dB SNR.

Expected impact: high. Effort: medium.

#### Priority 5: Candidate generation and automatic selection

- Generate multiple takes per turn.
- Rank with metrics and select best.
- Use ASR and speaker embeddings.

Expected impact: high. Effort: medium-high.

#### Priority 6: Engine diversification

- Compare Azure, Google Chirp 3 HD, and ElevenLabs on the same Hebrew scripts.
- Use engine-specific strengths.
- Avoid mixing engines within a speaker identity.

Expected impact: medium-high. Effort: medium-high.

#### Priority 7: Micro-prosody post-processing

- Add subtle breath/creak/tremor only after basic quality is fixed.
- Avoid global jitter/shimmer effects.

Expected impact: medium. Effort: high and risky.

### Proposed Production Configuration

```yaml
sample_rate: 16000
channels: 1
bit_depth: 16
format: wav

synthesis:
  unit: speaker_turn
  candidates_per_turn: 3
  max_turn_duration_s: 18
  max_ssml_chars: 350
  azure:
    voices:
      male: he-IL-AvriNeural
      female: he-IL-HilaNeural
    use_express_as: false
    rate_pct_range_default: [-6, 10]
    pitch_pct_range_default: [-4, 6]
  google_chirp3:
    language_code: he-IL
    use_preview_ssml_only_after_endpoint_test: true
    speaking_rate_range: [0.92, 1.12]

preprocessing:
  denoise_clean_tts: false
  lowpass_default: false
  highpass_hz: 60
  trim_leading_trailing_silence: true
  fade_ms: 8

quality_gates_turn:
  max_voiced_run_s: 2.8
  max_peak_dbfs: -0.1
  male_median_f0_hz_default: [80, 180]
  female_median_f0_hz_default: [150, 290]
  max_unexplained_f0_drift_st: 2.0
  asr_cer_clean_max: 0.15
  speaker_embedding_cosine_min: 0.74

scene:
  rir_source_mix:
    real_rir_probability: 0.65
    simulated_rir_probability: 0.35
  rt60_s_distribution:
    apartment_office: [0.25, 0.70]
    kitchen_hard_room: [0.45, 0.90]
  source_mic_distance_m: [0.7, 3.5]
  phone_height_m: 0.75

noise:
  datasets:
    - MUSAN
    - DEMAND
    - FSD50K
  snr_db_distribution:
    quiet: [25, 40, 0.15]
    normal: [15, 30, 0.50]
    noisy: [8, 18, 0.25]
    adverse: [0, 10, 0.10]

codec:
  probability: 0.20
  types:
    opus_24k: 0.50
    opus_12k: 0.25
    amr_nb: 0.15
    gsm: 0.10

final_loudness:
  integrated_lufs_range: [-26, -18]
  peak_dbfs_max: -1.0
```

### Recommended Validation Experiment

Run a controlled ablation on 100 Hebrew dialogue clips:

1. Baseline current pipeline.
2. Remove denoise + fix low-pass only.
3. Add turn-level synthesis + crossfades.
4. Add F0/rate quality gates.
5. Add room/device/noise simulation.
6. Add candidate generation/selection.

For each condition collect:

- human MOS naturalness on 1-5 scale;
- human “synthetic obviousness” binary label;
- ASR CER/WER;
- UTMOS/NISQA/DNSMOS;
- spectral high-frequency ratio;
- F0 drift statistics;
- artifact rate per hour;
- classifier downstream performance on a real validation set.

The most important final metric is downstream performance on real audio. A synthetic dataset can sound subjectively good but still teach the classifier wrong cues if the augmentation distribution is mismatched.

---

## References and URLs

### Platform documentation

- Microsoft Azure AI Speech language and voice support: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
- Microsoft Azure SSML voice and sound customization: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
- Microsoft Azure Text to Speech overview: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech
- Microsoft Azure HD voices: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/high-definition-voices
- Google Cloud Text-to-Speech Chirp 3 HD: https://cloud.google.com/text-to-speech/docs/chirp3-hd
- Google Cloud Text-to-Speech voice types: https://cloud.google.com/text-to-speech/docs/list-voices-and-types
- ElevenLabs Hebrew TTS: https://elevenlabs.io/languages/hebrew
- Coqui XTTS docs: https://docs.coqui.ai/en/latest/models/xtts.html
- Bark GitHub: https://github.com/suno-ai/bark

### Speech perception, emotion, and prosody

- Banse, R., & Scherer, K. R. (1996). *Acoustic profiles in vocal emotion expression*. https://pubmed.ncbi.nlm.nih.gov/8851745/
- Scherer, K. R. (2003). *Vocal communication of emotion: A review of research paradigms*. https://www1.cs.columbia.edu/~julia/papers/2003_Scherer_SpeechComm.pdf
- Paeschke, A., & Sendlmeier, W. F. (2000). *Prosodic characteristics of emotional speech*. https://www.isca-archive.org/speechemotion_2000/paeschke00_speechemotion.html
- Laukka, P. (2004). *Vocal Expression of Emotion*. https://www.diva-portal.org/smash/get/diva2:165425/fulltext01.pdf
- Gobl, C., & Ní Chasaide, A. (2003). *The role of voice quality in communicating emotion, mood and attitude*. https://www.cs.columbia.edu/~julia/papers/gobl03.pdf
- Klatt, D. H., & Klatt, L. C. (1990). *Analysis, synthesis, and perception of voice quality variations among female and male talkers*. https://pubmed.ncbi.nlm.nih.gov/2231467/
- Mixdorff, H., & Amir, N. (2002). *The prosody of Modern Hebrew*. https://www.isca-archive.org/speechprosody_2002/mixdorff02_speechprosody.html
- Silber-Varod, V., et al. (2016). *The acoustic correlates of lexical stress in Israeli Hebrew*. https://cris.tau.ac.il/en/publications/the-acoustic-correlates-of-lexical-stress-in-israeli-hebrew
- Pépiot, E. (2014). *Male and female speech: mean f0, f0 range, phonation type and speech rate*. https://www.isca-archive.org/speechprosody_2014/pepiot14_speechprosody.html

### Room/device/noise/audio tools

- pyroomacoustics: https://pyroomacoustics.readthedocs.io/
- gpuRIR: https://github.com/DavidDiazGuerra/gpuRIR
- OpenSLR MUSAN: https://www.openslr.org/17/
- DEMAND noise database: https://zenodo.org/record/1227121
- FSD50K: https://zenodo.org/record/4060432
- MIT IR Survey: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
- OpenAIR: https://www.openair.hosted.york.ac.uk/
- EchoThief: http://www.echothief.com/
- BUT ReverbDB: https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database
- OpenSLR RIRS_NOISES: https://www.openslr.org/28/
- audiomentations: https://github.com/iver56/audiomentations
- torch-audiomentations: https://github.com/asteroid-team/torch-audiomentations
- WavAugment: https://github.com/facebookresearch/WavAugment
- SoX: https://sox.sourceforge.net/sox.html
- FFmpeg: https://ffmpeg.org/

### Datasets and quality metrics

- RAVDESS paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
- RAVDESS dataset: https://zenodo.org/records/1188976
- IEMOCAP: https://sail.usc.edu/iemocap/
- IEMOCAP paper: https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf
- ESD: https://hltsingapore.github.io/ESD/
- ESD paper: https://arxiv.org/abs/2105.14762
- LibriTTS: https://www.openslr.org/60/
- LibriTTS paper: https://arxiv.org/abs/1904.02882
- LibriTTS-R: https://www.openslr.org/141/
- VCTK: https://datashare.ed.ac.uk/handle/10283/3443
- Mozilla Common Voice: https://commonvoice.mozilla.org/
- SASPEECH Hebrew corpus: https://www.openslr.org/134/
- HebTTS: https://github.com/slp-rl/HebTTS
- Phonikud: https://github.com/thewh1teagle/phonikud
- ITU-T P.862 PESQ: https://www.itu.int/rec/T-REC-P.862
- ITU-T P.863 POLQA: https://www.itu.int/rec/T-REC-P.863
- DNSMOS / DNS Challenge: https://github.com/microsoft/DNS-Challenge
- NISQA: https://github.com/gabrielmittag/NISQA
- UTMOSv2: https://github.com/sarulab-speech/UTMOSv2
