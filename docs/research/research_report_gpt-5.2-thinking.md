# Realistic Synthetic Speech Audio Generation for Hebrew — Comprehensive Research Report (GPT-5.2 Thinking)

**Context (your pipeline):** Hebrew multi-speaker dialogue clips (2–5 minutes) generated via cloud neural TTS (Azure Neural TTS; Google Cloud TTS Chirp 3 HD) with SSML/prosody control, then post-processed to emulate real recording conditions. Output is **16 kHz, mono, 16-bit PCM WAV**.

**Observed failure modes (as provided):**
- **Muffled / muddy** timbre (aggressive low-pass at ~7.5 kHz at 16 kHz SR; Wiener denoising applied to already-clean TTS).
- **Unnatural pitch behavior** (male F0 rises with anger; upward drift across clip; female F0 occasionally unrealistically low).
- **Robotic pacing** (too slow and too even, especially at low intensity).
- **TTS artifacts** (sustained vowels ~16 s; clicks at sentence boundaries; “voice identity” shifts when switching expressive styles).
- **No spatial realism** (sounds like studio / direct feed, not “phone on table in a room”).

This report is organized exactly per your requested structure and aims to be operational: specific tools, parameters, concrete recipes, and code.

---

## Part 1: How Humans Perceive Speech Naturalness

### 1.1 What acoustic features make speech sound “real” vs “synthetic”?
Human judgments of “naturalness” (often measured as MOS, Mean Opinion Score) are driven by *coherence across multiple cues* rather than any single feature. The largest practical drivers, especially for “obviously synthetic,” tend to be:

#### 1.1.1 Prosodic realism (timing + pitch + intensity)
- **Timing variability (“microtiming”)**:
  - Natural speech has non-uniform inter-word and inter-phrase timing; syllable durations vary by stress, position, and emotion.
  - Synthetic speech often has overly constant syllable timing and pause placement (especially when generated from plain punctuation without discourse-aware phrasing).
- **F0 (pitch) contour realism**:
  - Natural F0 contours show phrase-level declination, prominence on stressed syllables, and emotion-dependent *range expansion* (not always mean shift).
  - Synthetic speech often shows monotone segments, unnaturally smooth contours, or drift.
- **Intensity dynamics**:
  - Natural speech has meaningful RMS and spectral-energy changes with emphasis and emotion.
  - Synthetic speech often “turns up volume” without the spectral changes of real loud/pressed phonation.

**Key practical insight:** listeners are sensitive to *inconsistency* (e.g., “angry” content with a polite prosody; high loudness without high-frequency energy; an “angry male” realized mostly as higher mean F0 rather than increased range/energy).

#### 1.1.2 Voice quality realism (phonation / timbre)
Voice quality is heavily influenced by:
- **Spectral tilt** (how quickly high-frequency harmonics roll off). Breathy/soft voices have steeper tilt; tense/pressed voices have flatter tilt (more high-frequency energy). Studies on voice quality note that breathy voice is often associated with higher spectral tilt and lower harmonicity measures (e.g., HNR/CPP) (Patman, 2025; see: https://pure.york.ac.uk/portal/files/129149770/10.1515_phon-2025-0007.pdf).
- **Jitter / shimmer** (cycle-to-cycle F0/amplitude irregularity). Too little can sound “robotic”; too much sounds pathological.
- **Breathiness / aspiration noise** (low-level turbulent noise around the harmonics).
- **Creak / vocal fry** (irregular low-frequency phonation), common in stressed or low-arousal speech for some speakers/contexts.

A classic review of vocal quality types (modal/fry/falsetto/breathy) and their acoustics is Childers & Lee (1991): https://linguistics.berkeley.edu/~kjohnson/LSA317/Childers%26Lee1991.pdf

#### 1.1.3 Disfluencies and interactional cues
Real conversational speech includes:
- Filled pauses (“אממ”, “אה…”, “כאילו…”, “רגע…”), repairs, restarts.
- Turn-taking overlap and backchannels (“כן”, “נו”, “מה?”, “שנייה”).
Many high-end conversational TTS systems now intentionally model some disfluencies; but if you generate fully fluent, evenly paced dialogue, it will read as “scripted.”

#### 1.1.4 Recording-channel realism (room + device)
Even perfect “studio” speech sounds synthetic if you label it as “phone on a table.” Humans infer realism from:
- Room early reflections and late reverb (even subtle).
- Noise floor and stationary ambience (HVAC, distant traffic).
- Device/mic frequency response (band-limited or “presence peak”).
- Codec artifacts (AMR/Opus quantization, pre-echo, bandwidth switching).

### 1.2 Psychoacoustic research on speech naturalness perception (key papers)
There are two relevant literatures:

#### 1.2.1 Speech quality / naturalness evaluation
- **MOS and listening tests**: Human listening tests remain the gold standard; objective metrics correlate imperfectly as TTS improves.
- Tutorials and reviews summarize subjective vs objective evaluation (e.g., the ISCA/VoiceMOS tutorial slides include citations and context on modern MOS predictors like MOSNet/UTMOS/DNSMOS; see: https://voicemos-challenge-2023.github.io/speech-synthesis-evaluation/IS2025_tutorial.pdf).

Key takeaways:
- Intrusive metrics like PESQ were designed for telecommunications degradations, not expressive TTS.
- Neural MOS predictors (MOSNet, UTMOS) are useful as *quality gates*, but require calibration to your domain (Hebrew; your post-processing).

#### 1.2.2 Voice quality perception
- Voice quality categories and perceptual correlates are linked to spectral tilt, harmonicity, and noise measures (Childers & Lee, 1991; Patman, 2025).
- “Timbre” effects on intelligibility and listening effort are actively studied (e.g., work on vocal timbre and intelligibility: https://pubs.aip.org/asa/jel/article/5/7/075204/3356190/How-vocal-timbre-impacts-word-identification-and).

### 1.3 The role of F0 contour, jitter, shimmer, breathiness, spectral tilt
#### 1.3.1 F0 contour
Key perceptual cues:
- **Mean F0**: baseline pitch.
- **F0 range / SD**: expressiveness and emotional activation.
- **Contour shape**: declination, rises/falls at phrase ends, accentual peaks.

Emotion research often finds that **activation/arousal** correlates with higher mean F0 and larger F0 range (Schröder, 2001): https://www.isca-archive.org/eurospeech_2001/schroder01_eurospeech.pdf

#### 1.3.2 Jitter and shimmer
- Jitter: short-term variability in period length.
- Shimmer: short-term variability in amplitude.
In synthesis, “too clean” periodicity can sound artificial. Adding small controlled jitter/shimmer can help, but doing this naively at waveform level can introduce audible artifacts.

#### 1.3.3 Breathiness
Breathy voice tends to show:
- Higher spectral tilt.
- Lower harmonic-to-noise ratio / cepstral peak prominence (CPP).
(See Patman, 2025, and voice quality literature.)

#### 1.3.4 Spectral tilt (and “muffled/muddy” failure)
- A steeper tilt (less HF energy) sounds soft/distant/muffled.
- A flatter tilt (more HF energy) sounds tense/bright/effortful.
If you apply a steep low-pass near Nyquist **and** apply denoising that suppresses fine spectral detail, you effectively *increase perceived tilt* and reduce consonant clarity.

### 1.4 How emotion manifests acoustically in real speech (anger, fear, distress, calm)
There is large speaker and culture variability, but robust patterns exist.

#### 1.4.1 Broad patterns (well-supported)
- Active/high-arousal emotions (anger, panic fear) typically show:
  - Higher mean F0 and larger F0 range.
  - Higher intensity.
  - Faster articulation; shorter pauses.
  - Flatter spectral slope (more HF energy; “tenser” voice).
  - More abrupt F0 changes on stressed syllables.
These are explicitly summarized in classic vocal emotion work (Banse & Scherer, 1996) and dimensional modeling (Schröder, 2001).
- Banse & Scherer (1996), “Acoustic Profiles in Vocal Emotion Expression” (JPSP): https://www.columbia.edu/~rmk7/HC/HC_Readings/Scherer.pdf
- Schröder (2001): https://www.isca-archive.org/eurospeech_2001/schroder01_eurospeech.pdf

#### 1.4.2 Practical parameter ranges (usable for engineering)
Absolute ranges depend on speaker baseline (gender, physiology). A practical engineering approach is to specify:
- **mean F0 multiplier** relative to baseline,
- **F0 range multiplier** relative to baseline,
- **speaking rate multiplier** relative to baseline,
- **spectral tilt adjustment** (EQ shaping).

**Baseline adult speaking F0 (broad norms):**
- Men: average speaking F0 commonly **100–146 Hz** (Gelfer, 2005) and typical fundamental range often cited around **~90–155 Hz** (e.g., summary with sources in Baken, 2000; referenced via Wikipedia’s sourced section).
  - Gelfer (2005): https://www.sciencedirect.com/science/article/abs/pii/S0892199704001729
  - Summary including Baken (2000) citation: https://en.wikipedia.org/wiki/Voice_frequency
- Women: average speaking F0 commonly **188–221 Hz** (Gelfer, 2005), with typical fundamental often cited **~165–255 Hz** (see above).

**Emotion deltas (engineering-friendly, relative to baseline):**
- **Anger (hot / high arousal):**
  - Mean F0: +5% to +20% (can be *zero* for some male speakers; the bigger effect is often range/energy).
  - F0 range/SD: +20% to +60%.
  - Speaking rate: +5% to +25%.
  - Intensity: +3 to +8 dB RMS.
  - Spectral slope: flatter (more 2–6 kHz energy).
- **Fear / distress (high arousal, low control):**
  - Mean F0: +10% to +30% commonly.
  - F0 range: +30% to +80%.
  - Speaking rate: +10% to +35% (fear can be fast; distress can alternate fast bursts with longer pauses).
  - Breathiness can increase (depending on subtype).
- **Calm / neutral:**
  - Mean F0: baseline or slightly lower.
  - Range: smaller.
  - Rate: moderate; longer pauses; smoother contours.

**Important note (do not skip):** These ranges are *guidelines* rather than universal truths. In practice you should:
1) estimate per-voice baseline F0 + speech-rate from “neutral” TTS,
2) apply emotion controls primarily by *range + intensity + timing*,
3) constrain mean-F0 shifts so you do not create “male pitch rising with anger” in an unrealistic way (see Part 4).

### 1.5 Hebrew-specific prosody patterns
Hebrew differs from English in several ways that matter for TTS realism.

#### 1.5.1 Lexical stress and its acoustic correlates
Israeli Hebrew has lexical stress, with strong distributional patterns, and stress is realized via combinations of duration, intensity, and F0 cues (Silber-Varod, 2016):
https://www.sciencedirect.com/science/article/abs/pii/S0095447016000048

Also see additional Hebrew stress/prosody materials by Silber-Varod and colleagues (example lecture/notes: https://www.openu.ac.il/personal_sites/vered-silber-varod/download/Speech%20prosody_Silber-Varod%20and%20Green_11.10.12.pdf).

**Engineering implications for Hebrew TTS:**
- Stressed syllables often carry:
  - longer duration (vowel lengthening),
  - higher intensity,
  - local F0 movement (accent).
- If your SSML “rate” is globally slowed, you risk flattening stress cues and making Hebrew sound non-native.

#### 1.5.2 Prosodic boundaries and intonation units
Prosody segments speech into intonation units and boundary tones provide cues to segmentation and sentence type. Hebrew prosodic boundary patterns are studied in Speech Prosody proceedings and related work (e.g., Speech Prosody 2008 listing includes Hebrew boundary patterns): https://www.isca-archive.org/speechprosody_2008/

**Engineering implications:**
- Use *prosodic* phrasing, not just punctuation:
  - avoid long flat clauses,
  - add minor boundaries where Hebrew speakers naturally chunk,
  - vary pause lengths (comma vs sentence vs discourse boundary).

#### 1.5.3 Rhythm class and timing
The “stress-timed vs syllable-timed” framing is debated and not always stable across modern languages; for engineering, the practical point is:
- Hebrew tends to have strong stress effects and less extreme vowel reduction than English, so *timing and stress placement* are very noticeable.

#### 1.5.4 Hebrew speaking-rate reference values
A concrete Hebrew reference from Amir (2016/2015-ish preliminary observation) reports:
- speaking rate ~**5.90 syllables/sec**, articulation rate ~**6.42 syllables/sec** for professional Hebrew newscasters (not fully representative of all speakers, but a usable anchor):
https://www.researchgate.net/profile/Ofer-Amir/publication/303859712_Speaking_Rate_among_Adult_Hebrew_Speakers_A_Preliminary_Observation/links/5759045308ae414b8e3f6452/Speaking-Rate-among-Adult-Hebrew-Speakers-A-Preliminary-Observation.pdf

**Engineering implication:** Your “neutral calm” dialogue likely sounds slow if you are below ~4.5 SPS sustained; your low-intensity speech sounding too slow is consistent with this.

### 1.6 What makes recorded phone audio sound different from studio audio?
#### 1.6.1 Bandwidth and frequency shaping
- Traditional “voice band” telephony is ~**300–3400 Hz** (narrowband). Wideband telephony extends up to ~**7000 Hz**.
  - Voice-band summary: https://en.wikipedia.org/wiki/Voice_frequency
- AMR-WB (HD Voice, ITU G.722.2) provides ~**50–7000 Hz** bandwidth and is sampled at 16 kHz.
  - AMR-WB overview and bandwidth: https://www.loc.gov/preservation/digital/formats/fdd/fdd000255.shtml

#### 1.6.2 Device/mic response and handling
Smartphone microphones have their own response curves and often include:
- High-pass filtering (reduce handling/rumble).
- Midrange presence shaping (intelligibility).
- AGC / limiter behavior in some apps.
Research compares smartphone recordings to “gold standard” recording and discusses frequency response characteristics (example: Awan et al., 2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10545813/ (note: some environments may block automated access; use browser if needed).

#### 1.6.3 Room acoustics + device placement
- Phone on table introduces a strong early reflection from the tabletop (comb filtering) and changes directivity.
- Typical domestic/office rooms have modest RT60 values (often ~0.2–0.6 s), with strong early reflections.

#### 1.6.4 Codec artifacts
Phone/VoIP audio is often encoded with:
- AMR-NB / AMR-WB (cellular),
- Opus (VoIP, many apps),
- AAC/HE-AAC (some recordings),
- sometimes variable bandwidth switching.

Artifacts include:
- bandwidth limitation,
- quantization noise,
- transient smearing,
- pre-echo and “warble” at low bitrates.

---

## Part 2: State of the Art in Neural TTS Naturalness

### 2.1 Azure Neural TTS capabilities and limitations (he-IL AvriNeural / HilaNeural)
#### 2.1.1 Voice availability (current)
Azure’s official language/voice support list includes he-IL voices: `he-IL-HilaNeural` and `he-IL-AvriNeural` (Microsoft documentation):
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support

Microsoft’s Azure AI Speech blog has referenced quality updates including these he-IL voices (example update post includes he-IL voices among quality improvements):
https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/azure-ai-speech-text-to-speech-feb-2025-updates-new-hd-voices-and-more/4387263

#### 2.1.2 SSML control surface (what you can and cannot control)
Azure’s SSML implementation supports:
- `<voice>` selection,
- `<prosody>` (pitch/range/rate/volume),
- Azure extensions like `<mstts:express-as>` and `<mstts:silence>` (availability depends on voice/model family).
SSML overview and structure:
- https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup
- SSML structure: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure

**Prosody constraints (official):**
Azure documents explicit operational ranges:
- `pitch`: should be within **0.5× to 1.5×** original; can be specified as Hz, semitones (`st`), or percent.
- `rate`: within **0.5× to 2×** original; can be multiplier or percent.
(See official “Customize voice and sound with SSML” prosody section):
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice

**Engineering guidance from this:** do not attempt extreme pitch/rate values; they are often clipped, substituted, or produce artifacts.

#### 2.1.3 Known limitations relevant to your symptoms
- **Style switching artifacts / identity drift**: In many neural TTS systems, expressive style is implemented as a conditioning signal; aggressive switching between styles can create discontinuities or perceived “voice identity” shifts (especially across short turns). Azure’s docs emphasize that not all voices support all SSML tags, and behavior can differ by model family.
- **Long-form stability**: Microsoft has noted ongoing work on “consistency—particularly when processing lengthy or complex material” in HD updates (Azure Speech blog):
https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/azure-speech-%E2%80%93-neural-hd-text-to-speech-recent-voice-updates/4505380

### 2.2 Google Cloud TTS Chirp 3 HD for Hebrew
#### 2.2.1 Hebrew support and available voices
Google’s supported voices list includes Hebrew (he-IL) with Chirp3-HD voice names (official list page):
https://docs.cloud.google.com/text-to-speech/docs/list-voices-and-types

The same page shows specific Hebrew Chirp3-HD voices under he-IL (e.g., `he-IL-Chirp3-HD-Achernar`, etc.). (Scroll to the he-IL section on that page.)

#### 2.2.2 Control surface and SSML behavior: important nuance
Google’s Chirp 3 HD documentation includes:
- **Voice controls** such as pace control, with `speaking_rate` values **0.25–2.0** (official):
https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
- **SSML support section** stating that certain SSML tags (including `<prosody>`) are supported for synchronous requests, while SSML isn’t supported for streaming (same page).
However, Google’s voice list page includes a note stating Chirp 3 HD voices don’t support SSML input and pitch/rate-audio parameters. This appears inconsistent across docs and/or voice families; empirically, you should treat this as **“verify in your endpoint and request type”**.

**Practical recommendation:** implement a capability probe in your pipeline (send a minimal SSML request; check whether SSML is honored) and route to either:
- “SSML path” (if honored), or
- “voice-controls path” using `speaking_rate` and other supported controls (if SSML ignored).

### 2.3 What SSML prosody parameters do in neural TTS (under the hood)
In modern neural TTS, SSML parameters typically influence:
- **Duration model / aligner** (rate, breaks, emphasis),
- **Pitch predictor** or post-net scaling (pitch/range),
- **Energy predictor / loudness scaling** (volume),
- **Style embeddings** (express-as / style tokens).

They are rarely direct DSP transformations on waveform; instead they are (a) constraints/conditioning during acoustic feature generation, then (b) vocoding to waveform. This is why:
- extreme settings can push the model out of distribution,
- interactions (pitch + rate + volume) can produce non-linear effects.

### 2.4 Known issues with Azure `mstts:express-as`
There is no single official “bug list” for expressive styles, but community reports and Q&A indicate:
- Differences between Azure studio tools and API behavior for certain style applications (example discussion around whisper style application to small spans):
https://learn.microsoft.com/en-us/answers/questions/1863460/issue-with-the-azure-text-to-speech-%28tts%29-service

**Engineering implication:** treat `express-as` usage as “supported but fragile” at fine granularity; prefer turn-level usage.

### 2.5 SSML `<prosody>` interactions (pitch + rate + volume)
Common failure patterns:
- Reducing `rate` too much can cause unnatural pauses and “robotic drawl.”
- Increasing `pitch` while also increasing `volume` can produce an unnatural “shouty squeak” for male voices.
- Combining large negative pitch shifts with slow rate can produce “female voice too low” or creaky artifacts.

**Best practice:** keep prosody changes within moderate bounds; use style tokens for emotion whenever possible, and use prosody mainly for small adjustments.

### 2.6 Best practices for SSML prosody control (practically useful)
#### 2.6.1 Parameter ranges that usually remain natural (starting defaults)
For conversational dialogue (turn-level):
- `rate`: **0.90–1.15** (rarely go below 0.85; rarely above 1.25).
- `pitch`: **-5% to +5%** for male voices; **-7% to +7%** for female voices (if needed).
- `range`: increase for high-arousal: **+10% to +35%**; decrease for calm: **-10% to -25%**.
- `volume`: use dB if supported; keep changes to **±3 dB** for “natural,” **+6 dB** for “raised voice” (but combine with spectral shaping, not only RMS).

Azure explicitly documents pitch/rate operational bounds (0.5–1.5× pitch, 0.5–2× rate), but those bounds include many unnatural regions; use the moderate ranges above unless you have evidence otherwise.

#### 2.6.2 Prosody shaping strategy for emotion (robust)
- Use **style tokens** (express-as) for coarse emotion.
- Use `range` to expand F0 variability rather than raising baseline for male anger.
- Use pauses and phrasing (breaks, sentence boundaries) to create believable escalation rather than just pitch shifting.

### 2.7 How professional voice-over / audiobook producers use TTS with markup (practical tricks)
Common production patterns:
- **Chunking**: render in sentence/paragraph chunks, then assemble with controlled crossfades and room/ambience continuity.
- **Lexicons**: enforce correct pronunciation of names, slang, acronyms.
- **Breath/pause design**: add micro-pauses and breaths around long clauses; avoid uniform punctuation pauses.
- **Consistency first**: prefer subtle prosody controls plus post-processing, rather than extreme expressive tags that can drift identity.
- **Manual review loops**: automated gates + spot listening on samples per batch.

### 2.8 Comparison: Azure vs Google vs ElevenLabs vs Coqui/XTTS vs Bark (and others)
This section focuses on what is *verifiable* and what tends to matter for Hebrew emotional dialogue.

#### 2.8.1 Cloud engines with explicit Hebrew support
- **Azure Speech**: officially supports Hebrew voices `he-IL-HilaNeural` and `he-IL-AvriNeural` and supports rich SSML controls.
  - Voice availability: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
  - SSML prosody specs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
- **Google Cloud TTS Chirp3-HD**: official voice list includes many he-IL Chirp3-HD voices.
  - Voice list: https://docs.cloud.google.com/text-to-speech/docs/list-voices-and-types
  - Chirp3-HD overview and pace: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd

**Operational difference:** Google’s Chirp3-HD exposes an explicit `speaking_rate` control of **0.25–2.0** (on the Chirp3-HD page). Azure’s rate control is documented as **0.5–2.0** and has multiple representations.

#### 2.8.2 Open-source engines (Hebrew support variability)
- **Coqui XTTS v2**: community issues suggest Hebrew support is not first-class in XTTS v2 (example request to add Hebrew):
  https://github.com/coqui-ai/TTS/issues/3512
- **Bark**: multilingual, can generate conversational speech and non-speech sounds; quality and stability vary; see Coqui docs referencing Bark:
  https://coqui-tts.readthedocs.io/en/latest/models/bark.html
For high-quality Hebrew emotional speech, open-source options often require custom training or voice conversion pipelines; plan accordingly.

---

## Part 3: Post-Processing for Realism

### 3.1 Room acoustics simulation
You have two broad approaches:

#### 3.1.1 Convolution with real RIRs (recommended baseline)
Use measured room impulse responses (RIRs) and convolve your dry TTS with them:
- Libraries and datasets:
  - OpenAIR IR library: https://www.openair.hosted.york.ac.uk/
  - BUT ReverbDB (real RIRs + noise): https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
  - OpenSLR RIR & Noise database (SLR28): https://www.openslr.org/28/
  - A curated list of public RIR databases: https://github.com/Graphi07/room-impulse-responses
  - RealRIRs loader library: https://github.com/jonashaag/RealRIRs

**Advantages:** realistic frequency-dependent reflections, device-mic placement diversity (if dataset provides), avoids simulation mismatch.

#### 3.1.2 Simulated RIRs (for controllability and scale)
If you need systematic control and large volume:
- **pyroomacoustics** (image-source method + optional ray tracing):
  - Docs: https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
  - GitHub: https://github.com/LCAV/pyroomacoustics
  - The code notes suggest hybrid ISM+ray tracing with `max_order=3` as a practical recommendation (see pyroomacoustics `room.py` comments and examples).
- **gpuRIR** (GPU-accelerated image-source simulation):
  - GitHub: https://github.com/DavidDiazGuerra/gpuRIR
  - Paper: Diaz-Guerra et al., 2018/2020: https://arxiv.org/abs/1810.11359

**ISM vs ray tracing (quick engineering view):**
- ISM captures early specular reflections well; may under-model late diffuse tail unless many orders.
- Ray tracing better approximates late reverberation and scattering; more parameters.

#### 3.1.3 Realistic parameter ranges for small rooms (apartment/office)
You should explicitly model:
- Room dimensions: e.g., **(3–6 m) × (3–8 m) × (2.4–3.0 m)**.
- Source-to-mic distance: **0.3–2.5 m**.
- RT60:
  - Small treated office: ~**0.2–0.35 s**
  - Typical furnished room: ~**0.3–0.6 s**
  - Hard surfaces / empty room: can exceed **0.7 s**
A concrete example of measured RT60 variants in a controlled room dataset: RWTH Multi-Channel Impulse Response Database reports scenarios with RT60 **160 ms, 360 ms, 610 ms**:
https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/

**Recommendation:** start your synthetic room augmentation with RT60 in **0.25–0.55 s**, and later expand.

#### 3.1.4 Early reflections vs late reverb, and DRR
- Direct-to-reverberant ratio (DRR) strongly affects perceived distance.
- Phone-on-table in a room usually has a strong direct path but also strong early reflections (table + nearby wall).
If you only add a smooth reverb tail (like a music reverb), it won’t feel like “phone in room.”

### 3.2 Device/channel simulation (phone mic, phone-on-table pickup)
#### 3.2.1 Frequency response shaping (practical EQ)
Smartphones commonly:
- attenuate very low frequencies (HPF),
- emphasize intelligibility band (~1.5–4 kHz),
- may roll off the very top (depending on SR and pipeline).

At 16 kHz sampling, Nyquist is 8 kHz. You generally want to preserve energy up to ~7.5–7.8 kHz, but not apply a steep brick-wall filter that dulls consonants.

**Suggested phone-mic EQ (starting curve for 16 kHz audio):**
- High-pass: 80–120 Hz (2nd order).
- Presence boost: +2 to +4 dB around 2.5–3.5 kHz (Q ~0.8–1.2).
- Gentle high-shelf: -1 to -3 dB above ~6.5 kHz (very gentle), *not* a hard low-pass at 7.5 kHz.
- Optional: mild mid cut -1 to -2 dB around 250–400 Hz if muddy.

#### 3.2.2 Phone-on-table “two-path” model
A phone on a table creates a strong first reflection from the tabletop:
- Delay: ~0.3–2.0 ms (depends on geometry).
- Attenuation: -3 to -12 dB relative to direct.
This creates comb filtering that is very recognizable as “device near surface.”

**Implement:** add a delayed, low-pass filtered copy of the signal:
- `y = x + a * LPF(x delayed by d)`
- with `d` in samples corresponding to 0.5–1.5 ms at 16 kHz (8–24 samples).

### 3.3 Background noise and ambience
#### 3.3.1 Ambient noise profiles to use
- Domestic: HVAC + distant traffic + refrigerator cycles.
- Office: HVAC + keyboard/mouse + distant voices.
- Clinic room: HVAC + distant corridor.

Use real noise recordings when possible. For open datasets, OpenSLR SLR28 includes noise recordings in addition to RIRs: https://www.openslr.org/28/

#### 3.3.2 Realistic SNR ranges (engineering values)
For “phone on table”:
- Quiet room: **20–35 dB SNR**
- Typical room: **10–25 dB**
For “phone in pocket / far from speakers”:
- **0–15 dB** is plausible, often with muffling and occlusion artifacts.

**Recommendation:** don’t randomize SNR uniformly. Use a mixture distribution (e.g., 60% in 15–30 dB, 30% in 8–15 dB, 10% in 0–8 dB) depending on your target use case.

### 3.4 Reverberation: RT60, early reflections, DRR
As above, RT60 for small rooms often ~0.2–0.6 s. The perception of “distance” also depends on DRR:
- Near-field: high DRR (direct dominates).
- Far-field: lower DRR; early reflections become prominent.

**Recommendation:** sample DRR and early reflection patterns explicitly by varying:
- distance,
- mic directivity (even simple),
- source directivity (optional),
- room absorption.

### 3.5 Codec simulation (GSM/AMR/Opus artifacts)
Codec simulation is often essential if you want it to sound like “phone audio,” not “studio in a room.”

#### 3.5.1 What “phone-quality” commonly is
- Narrowband voice: ~300–3400 Hz.
- Wideband voice (HD Voice): up to ~7000 Hz (AMR-WB / G.722.2).
  - AMR-WB bandwidth summary: https://www.loc.gov/preservation/digital/formats/fdd/fdd000255.shtml

#### 3.5.2 Practical simulation using FFmpeg
**AMR-NB (8 kHz narrowband):**
```bash
ffmpeg -i in.wav -ar 8000 -ac 1 -c:a amr_nb -b:a 12.2k tmp.amr
ffmpeg -i tmp.amr -ar 16000 -ac 1 -c:a pcm_s16le out_amrnb_upsampled.wav
```

**AMR-WB (16 kHz wideband):**
```bash
ffmpeg -i in.wav -ar 16000 -ac 1 -c:a amr_wb -b:a 12.65k tmp.awb
ffmpeg -i tmp.awb -ar 16000 -ac 1 -c:a pcm_s16le out_amrwb.wav
```

**Opus (VoIP-like):**
```bash
ffmpeg -i in.wav -ar 48000 -ac 1 -c:a libopus -b:a 16k -application voip tmp.ogg
ffmpeg -i tmp.ogg -ar 16000 -ac 1 -c:a pcm_s16le out_opus_voip.wav
```

**Recommendation:** apply codec after room+noise (closer to reality: the call app encodes the captured signal).

### 3.6 Lowpass/highpass filtering for 16 kHz audio (what cutoffs matter)
With 16 kHz sampling:
- Nyquist = 8 kHz.
- Speech intelligibility relies heavily on 1–4 kHz; consonant clarity benefits from 4–8 kHz energy.

A low-pass at 7.5 kHz is not inherently wrong, but if it is steep or combined with denoising, it can remove the “sparkle” cues that make speech sound real. If you need low-pass:
- use gentle roll-off,
- keep cutoff near 7.8 kHz,
- avoid aggressive suppression around 5–7 kHz.

### 3.7 Denoising on clean audio: why Wiener denoising hurts clean TTS
Wiener denoising assumes a noisy observation and tries to suppress components considered noise-like. On already-clean TTS:
- it will suppress high-frequency frication detail,
- it can introduce “musical noise” or smear,
- it increases perceived muffling and reduces spectral contrast.

**Better alternatives:**
- Don’t denoise clean audio at all.
- If you add noise, denoise only under realistic conditions (and keep a raw noisy version too).
- If the goal is to reduce TTS hiss, use targeted spectral gating with conservative thresholds, or a gentle EQ shelf rather than full Wiener denoise.

---

## Part 4: Pitch and Prosody Engineering

### 4.1 Real angry vs calm speech in F0, rate, intensity, spectral tilt
Schröder (2001) reports that high activation emotions correlate with:
- higher mean and range of F0,
- shorter pauses,
- increased intensity,
- flatter spectral slope (more HF energy).
https://www.isca-archive.org/eurospeech_2001/schroder01_eurospeech.pdf

Banse & Scherer (1996) summarize classic findings:
- Anger: increased mean F0 and mean energy; often increased HF energy; articulation rate increases; “hot anger” shows larger F0 variability.
https://www.columbia.edu/~rmk7/HC/HC_Readings/Scherer.pdf

### 4.2 Male vs female pitch ranges (usable defaults)
Use per-speaker baselines, but as a practical anchor:
- Men: average speaking F0 ~100–146 Hz; Women: ~188–221 Hz (Gelfer, 2005):
https://www.sciencedirect.com/science/article/abs/pii/S0892199704001729
Other sources summarize wider typical ranges (with citations to clinical voice measurement):
https://en.wikipedia.org/wiki/Voice_frequency

**Hebrew-specific note:** published Hebrew-specific *adult baseline F0* norms are not as easily accessible as speaking-rate norms; do not assume Hebrew is fundamentally different from other languages in baseline F0—speaker physiology dominates. Treat “Hebrew-specific” primarily as **stress, phrasing, and intonation** differences rather than baseline pitch.

### 4.3 How pitch should change across an escalating argument (F0 contour patterns)
A realistic escalation is not “monotonic rising pitch.” Typical patterns:
- Increased **F0 range**: more frequent and steeper rises/falls; more prominent accents.
- Mean F0 may rise modestly, but especially for male voices, *stability of mean* plus range expansion often sounds more believable than large mean shifts.
- Increased **intensity** and **spectral brightness** with “pressed” voice quality.
- Faster speech with shorter pauses, but with occasional “control breaks” (interruptions, breaths).

**Engineering pattern (turn-level):**
- calm baseline: range = 1.0×, mean = 1.0×
- irritation: range = 1.2×, mean = 1.02×
- anger: range = 1.4×, mean = 1.05× (male) / 1.08× (female)
- shouting: range = 1.5×, mean = 1.05×, plus brightness boost, plus clipping/limiting behavior

### 4.4 Pitch LEVEL vs pitch VARIATION
- Pitch level: mean/median F0.
- Pitch variation: SD/range of F0.

**Your symptom (“male pitch rises with anger”)** is often an overuse of pitch-level control. For male anger, prefer:
- keep mean nearly stable (+0–5%),
- expand range (+20–60%),
- increase intensity/brightness.

### 4.5 Preventing pitch drift in multi-turn TTS rendering
Pitch drift across minutes often comes from:
- style conditioning state not resetting cleanly,
- long-context synthesis instability,
- cumulative prosody controls across turns.

**Mitigations:**
1) **Synthesize per-turn (or per-sentence) chunks**, not whole 2–5 minute clips in a single request.
2) **Reset style/prosody per chunk** (new `<voice>` element in Azure can also reset certain settings; `mstts:silence` explicitly notes resets via new voice elements).
3) **Measure F0 per chunk** (via `pyin`, `dio`, or `praat`) and enforce a target distribution:
   - clamp mean drift to within ±3–5% of per-speaker baseline for neutral/calm,
   - allow drift for fear/distress only if explicitly desired.

### 4.6 Rate manipulation: natural Hebrew speaking rates by emotion
Anchors:
- Hebrew newscaster speaking rate reported ~5.90 syll/s; articulation ~6.42 syll/s (Amir, preliminary):
https://www.researchgate.net/profile/Ofer-Amir/publication/303859712_Speaking_Rate_among_Adult_Hebrew_Speakers_A_Preliminary_Observation/links/5759045308ae414b8e3f6452/Speaking-Rate-among-Adult-Hebrew-Speakers-A-Preliminary-Observation.pdf

**Practical targets (heuristics for your TTS):**
- Calm: 4.3–5.2 SPS (longer pauses, lower articulation rate).
- Neutral conversation: 4.8–6.0 SPS.
- Anger: 5.5–7.0 SPS (shorter pauses; more bursts).
- Fear/distress: highly variable; often 5.5–8.0 SPS *in bursts* plus longer hesitation pauses.

**Important:** SSML rate controls are global scalars; they won’t create bursts naturally. You need pause design + chunk-level control.

### 4.7 Loudness and RMS: real shouting vs TTS “loud”
Real shouting differs by:
- more high-frequency energy (flattened spectral tilt),
- nonlinear vocal fold behavior,
- formant shifts and increased subglottal pressure,
- often mic overload / clipping / compression in phone recordings.

TTS “volume=loud” usually just increases amplitude, missing the spectral and nonlinear aspects. Therefore:
- use volume increase modestly,
- add brightness EQ and mild saturation/limiting,
- optionally add mild clipping in a controlled way to emulate phone AGC/limiter.

### 4.8 Adding micro-prosody (jitter, shimmer, creak, breathiness) in post
This is hard to do well without a vocoder or pitch-synchronous processing.
Options:
- **WORLD vocoder** (pyworld) to resynthesize with modified F0 and aperiodicity. Risk: robotic artifacts if overused.
- **Praat** (via parselmouth) for pitch manipulation and voice quality tweaks; careful.
- **Add low-level aspiration noise** modulated by voiced/unvoiced regions (safer than jitter).

**Practical recommendation:** prioritize room/device realism and prosody control before micro-prosody. Micro-prosody is a “last 10%” and can easily degrade quality.

---

## Part 5: Eliminating TTS Artifacts

### 5.1 Common Azure Neural TTS artifacts (and plausible causes)
#### 5.1.1 Sustained vowels / elongated phonemes (e.g., 16 seconds)
Likely causes:
- pathological SSML structure (unclosed tags, nested prosody),
- extremely slow rate or long pauses interacting with model,
- very long input chunk causing internal alignment failure.

Mitigation:
- enforce **max chunk length** (characters and duration) and synthesize in smaller segments.
- validate SSML XML and avoid exotic nesting.
- clamp `rate` away from the extremes; Azure allows 0.5–2× but avoid near 0.5 unless necessary.

### 5.1.2 Clicks at sentence boundaries
Likely causes:
- concatenation/joining without fades,
- discontinuities when switching styles or voices,
- codec/format conversions introducing frame boundary artifacts.

Mitigation:
- apply 5–20 ms fade-out/fade-in on every segment boundary,
- join at zero-crossings when possible,
- avoid switching `mstts:express-as` mid-sentence; prefer per-turn.

### 5.1.3 Voice identity shifts between turns when changing style
Likely causes:
- style embedding moves the speaker representation (common in expressive TTS),
- abrupt style changes create discontinuities.

Mitigation:
- keep each speaker’s voice identity fixed; vary emotion with moderate `styledegree` and minimal pitch-level change.
- if style must change, keep it within a small set and avoid rapid alternation.

### 5.2 SSML patterns that can cause failures
In Azure SSML docs, `mstts:silence` has specific limitations (only certain boundaries; resets require a new `<voice>` element):
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure

General risky patterns:
- very long `<break time="...">` inside dense sentences,
- nested prosody tags,
- aggressive combinations: `rate="0.5"` + large pitch shift + express-as.

### 5.3 Safe parameter ranges (Azure SSML prosody tags)
Azure’s official bounds:
- pitch within 0.5–1.5×; rate 0.5–2× (official):
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice

**Empirically safe ranges (recommended defaults):**
- `rate`: 0.90–1.15
- `pitch`: -5% to +5% (male); -7% to +7% (female)
- `range`: -20% to +35%
- `break`: 80–400 ms typical; up to 700 ms for major boundaries
- `mstts:silence Sentenceboundary`: 100–350 ms (plus natural silence)

### 5.4 Detect and automatically filter broken TTS output
Implement an automated “quality gate” stage after synthesis and after post-processing.

#### 5.4.1 Silence / duration sanity checks
- clip duration must be within expected (120–300 s).
- reject if:
  - >10% of clip is absolute silence below -60 dBFS,
  - or any single continuous “voiced” segment exceeds (say) 8–10 seconds without boundaries (detect via pitch tracker + VAD).

#### 5.4.2 Spectral anomaly detection (catch muffled/overfiltered)
Compute long-term average spectrum (LTAS) and check:
- energy ratio `E(4–7kHz) / E(0.3–4kHz)` within plausible bounds.
- if the ratio is extremely low (e.g., < -25 dB relative), the sample will sound muffled.

#### 5.4.3 Sustained vowel detection (your 16-second vowel case)
Heuristic:
- run F0 estimator (e.g., librosa.pyin) and spectral flux.
- flag if there exists a region > 3–5 seconds where:
  - voiced probability high,
  - spectral flux very low,
  - intensity stable.

#### 5.4.4 Click detection
Clicks appear as short high-energy broadband spikes.
- compute sample-wise derivative or short-window spectral flatness peaks,
- flag if too many spikes exceed threshold (e.g., > 10 spikes with peak > -3 dBFS and duration < 5 ms).

### 5.5 Crossfading and segment joining (eliminate boundary clicks)
**Best practice:** always fade edges of TTS chunks.

For two chunks `a` and `b`:
- apply `fade_out` last 10 ms of `a`,
- apply `fade_in` first 10 ms of `b`,
- overlap-add.

In FFmpeg:
```bash
ffmpeg -i a.wav -i b.wav -filter_complex "acrossfade=d=0.01" out.wav
```

In Python (numpy):
```python
import numpy as np

def crossfade(a, b, n=160):  # 10ms at 16kHz
    fade = np.linspace(0, 1, n, dtype=np.float32)
    a_tail = a[-n:] * (1 - fade)
    b_head = b[:n] * fade
    return np.concatenate([a[:-n], a_tail + b_head, b[n:]])
```

### 5.6 Voice consistency across turns when changing emotion/style
Add a **speaker-embedding consistency gate**:
- Extract embeddings using ECAPA-TDNN or x-vector.
- Compare each turn to that speaker’s reference embedding.
- Regenerate if similarity falls below a threshold (needs calibration; start with cosine similarity > 0.75–0.85 depending on embedding model and domain).

---

## Part 6: Existing Projects and Tools

### 6.1 Open-source projects doing similar work
#### 6.1.1 Audio augmentation for ML
- audiomentations (CPU): https://github.com/iver56/audiomentations
- torch-audiomentations (PyTorch): https://pypi.org/project/torch-audiomentations/
- WavAugment (Meta): https://github.com/facebookresearch/WavAugment
- torchaudio augmentation tutorial including codec simulation:
  https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html

#### 6.1.2 Room simulation
- pyroomacoustics: https://github.com/LCAV/pyroomacoustics
- gpuRIR: https://github.com/DavidDiazGuerra/gpuRIR
  - paper: Diaz-Guerra et al., 2018/2020: https://arxiv.org/abs/1810.11359

#### 6.1.3 RIR datasets
- OpenAIR: https://www.openair.hosted.york.ac.uk/
- BUT ReverbDB: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
- OpenSLR SLR28 (RIR+noise): https://www.openslr.org/28/
- EchoThief IR library (creative spaces): https://www.echothief.com/
- Curated list: https://github.com/Graphi07/room-impulse-responses
- RealRIRs loaders: https://github.com/jonashaag/RealRIRs

### 6.2 Relevant datasets and how they were created
#### 6.2.1 Emotional speech datasets
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song): acted emotions; speech audio-only WAV; commonly referenced as 48 kHz / 16-bit WAV (example dataset page):
  https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
- **IEMOCAP**: acted dyadic sessions, multimodal, multiple speakers; official site:
  https://sail.usc.edu/iemocap/
- **ESD (Emotional Speech Dataset)**: widely used; details vary by release; verify the source you use (some are on GitHub/HF). (Not all sources are stable; check the canonical paper/repo for your chosen ESD variant.)
- **LibriTTS / VCTK**: high-quality read speech / multi-speaker corpora; designed for TTS; typically curated and normalized. (Use the official corpus pages for exact versions.)

#### 6.2.2 General ASR speech datasets (Hebrew relevant)
- **Mozilla Common Voice**: crowdsourced multilingual speech corpus (Ardila et al., 2019): https://arxiv.org/abs/1912.06670
  Dataset downloads: https://commonvoice.mozilla.org/datasets
- **ivrit.ai**: large Hebrew speech dataset (Marmor et al., 2023): https://arxiv.org/abs/2307.08720
  Project page: https://www.ivrit.ai/en/ivrit-ai-2/
  Hugging Face org: https://huggingface.co/ivrit-ai
- **SASPEECH** (single-speaker Hebrew dataset; Interspeech 2023):
  OpenSLR listing: https://openslr.org/134/
  ISCA paper page: https://www.isca-archive.org/interspeech_2023/sharoni23_interspeech.html
- **HebDB** (weakly supervised Hebrew speech dataset):
  https://pages.cs.huji.ac.il/adiyoss-lab/HebDB/

### 6.3 Audio augmentation libraries: which are best for your use case?
For your use case (synthetic speech realism for classification), a good stack is:

- **audiomentations**: quick CPU augmentations (noise, gain, filtering, time stretch).
  https://github.com/iver56/audiomentations
- **torch-audiomentations**: GPU-friendly augmentation in PyTorch training loop.
  https://pypi.org/project/torch-audiomentations/
- **torchaudio**: signal effects, RIR convolution, and codec simulation workflows.
  https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
- **WavAugment**: augmentation set used in SSL speech representation learning (Kharitonov et al.).
  https://github.com/facebookresearch/WavAugment
- **SoX / FFmpeg**: robust offline effect chains and codec simulation.

**Recommendation:** Use FFmpeg for codec simulation and torchaudio/pyroomacoustics for convolution. Use audiomentations only for simple additive noise and gain shaping.

### 6.4 Speech quality assessment tools (PESQ, POLQA, DNSMOS, UTMOS)
#### 6.4.1 Metrics and licensing realities
- PESQ is ITU-T P.862; POLQA is ITU-T P.863 and is generally licensed (Opticom licensing overview):
  https://www.opticom.de/licensing/licensing.php
- A curated overview of methods (including PESQ, POLQA, VISQOL, DNSMOS) and openness notes:
  https://github.com/jonnor/machinehearing/blob/master/audio-quality/README.md
- Guidance that multiple PESQ versions/implementations differ significantly (important for reproducibility):
  https://arxiv.org/html/2505.19760v1

#### 6.4.2 Practical recommendation for your pipeline
Use a *bundle* of gates:
- **DNSMOS**-style predictors for distortions/noise (good for post-processing artifacts).
- **MOSNet / UTMOS** for “TTS naturalness” scoring, but calibrate with Hebrew samples.
- **Speaker embedding consistency** for identity drift.
- Simple signal heuristics for clicks, long vowels, muffling.

### 6.5 Hebrew-specific TTS or speech resources
- ivrit.ai (dataset + license): https://www.ivrit.ai/en/ivrit-ai-2/ and https://www.ivrit.ai/en/the-license/
- SASPEECH (single speaker, 44.1 kHz): https://openslr.org/134/
- HebDB: https://pages.cs.huji.ac.il/adiyoss-lab/HebDB/

---

## Part 7: Recommended Architecture Changes (Concrete)

### 7.1 Biggest wins first (highest ROI)
#### Priority 0 (Stop actively degrading quality)
1) **Remove Wiener denoising** on clean TTS.
   - Only use denoising when you *added noise*, and even then keep an un-denoised noisy version.
2) **Remove or soften the aggressive low-pass**.
   - If you must low-pass at 16 kHz, use gentle roll-off near 7.8 kHz (not a steep cutoff at 7.5 kHz), and avoid stacking with denoising.
3) **Stop using pitch-level shifts for anger on male voices**.
   - Use range and intensity instead; keep mean F0 stable.

#### Priority 1 (Add realism that dominates perception)
4) **Add room+device simulation** with either real RIR convolution or pyroomacoustics/gpuRIR, plus phone EQ and “table reflection.”
5) **Add background ambience** with realistic SNR distribution.
6) **Add codec simulation** (Opus voip; optionally AMR-WB for “cell call” flavor).

#### Priority 2 (Fix TTS generation stability)
7) **Chunked synthesis**: per-turn or per-sentence rendering + crossfade assembly.
8) **Voice consistency gate** using speaker embeddings.
9) **Automated artifact detection** for sustained vowels/clicks and regeneration.

### 7.2 Concrete pipeline architecture (recommended)
#### Stage A — Text + prosody planning (per turn)
Input: scripted dialogue with speaker IDs, emotion labels, intensity, and target scene metadata.

Output: for each turn:
- text chunk (ideally 1–2 sentences),
- desired prosody targets:
  - `style` (emotion),
  - `styledegree` (intensity),
  - `rate` multiplier,
  - `range` multiplier,
  - pause plan (pre/post, internal breaks).

#### Stage B — TTS rendering (per turn)
- Render each turn independently.
- Store metadata: chosen voice, SSML used, request params.

**Azure SSML example (turn-level):**
```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="he-IL">
  <voice name="he-IL-AvriNeural">
    <mstts:silence type="Sentenceboundary" value="180ms"/>
    <mstts:express-as style="angry" styledegree="1.0">
      <prosody rate="1.08" pitch="0%" range="+25%">
        די! תפסיק כבר. אני אומר לך שזה לא מקובל.
      </prosody>
    </mstts:express-as>
  </voice>
</speak>
```
Notes:
- Keep `pitch` near 0 for male anger; increase `range` instead.
- Keep `rate` modestly above 1 for anger.

**Google Chirp3-HD request example (pace control):**
Use `speaking_rate` in request as documented (0.25–2.0):
```json
{
  "audio_config": {"audio_encoding": "LINEAR16", "speaking_rate": 1.10},
  "input": {"text": "די! תפסיק כבר. אני אומר לך שזה לא מקובל."},
  "voice": {"language_code": "he-IL", "name": "he-il-Chirp3-HD-Achird"}
}
```
(Chirp3-HD docs: https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd)

If SSML is honored on your endpoint, you can use `<prosody>`; otherwise rely on the documented voice controls and keep SSML minimal.

#### Stage C — Assembly (crossfade, timing, overlaps)
- Add inter-turn gaps (with variation, not constant).
- Add overlaps/backchannels (optional).
- Apply crossfades (10 ms default, longer for some boundaries).

#### Stage D — Acoustic scene rendering (room + device + noise)
Recommended order:
1) Convolve speech with RIR (real or simulated).
2) Apply phone/device EQ (HPF + presence + gentle shelf).
3) Add ambience/noise at target SNR distribution.
4) Apply codec simulation (Opus voip for WhatsApp-like, AMR-WB for cellular).
5) Optional: mild limiter/clipping to emulate phone AGC.

#### Stage E — Quality gates and regeneration
Run gates after Stage B (per turn) and after Stage D (full clip).

### 7.3 Specific changes to remove/modify (your current pain points)
#### 7.3.1 Low-pass filter
- Replace steep lowpass at 7.5 kHz with:
  - either **no lowpass**, or
  - gentle lowpass at 7.8 kHz (6 dB/oct), or
  - high-shelf -1 to -2 dB above 6.5 kHz.

#### 7.3.2 Wiener denoising
- Remove entirely for clean TTS output.
- If you need denoising for noisy simulations, apply it only to those scenarios and keep a mix of denoised/non-denoised.

### 7.4 Post-processing additions (tools and parameter ranges)
#### 7.4.1 Room simulation (choose one baseline)
- **Real RIR baseline:** sample RIRs from SLR28 or BUT ReverbDB.
  - SLR28: https://www.openslr.org/28/
  - BUT ReverbDB: https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
- **Simulated baseline:** pyroomacoustics:
  - room dims sampled as above,
  - start with RT60 target 0.25–0.55 s,
  - hybrid simulation with ray tracing and `max_order=3` as suggested in docs.

#### 7.4.2 Phone-on-table model
- Add one early reflection:
  - delay: 0.6–1.2 ms (10–20 samples at 16 kHz)
  - attenuation: -6 to -10 dB
  - apply LPF to reflected copy (cut 4–6 kHz) to mimic surface losses.

#### 7.4.3 Background noise
- SNR distribution: concentrate in 15–30 dB; add tails down to 0–8 dB for “hard cases.”

#### 7.4.4 Codec simulation
- Always include at least Opus voip simulation for “phone realism.”
- Include AMR-WB for “cell call” variant (50–7000 Hz band).

### 7.5 SSML/prosody changes (concrete)
#### 7.5.1 Stop encoding anger as “higher mean pitch for male”
Instead:
- use express-as style=angry (if supported),
- increase `range` more than `pitch`,
- speed up `rate` slightly,
- shorten pauses and add interruptions.

#### 7.5.2 Add Hebrew-specific stress and pacing cues
- Use `<s>` sentence wrappers and punctuation-based phrasing, but also insert short `<break>` at discourse boundaries.
- Add some disfluencies in low-intensity or distressed states:
  - calm hesitation: “אממ…”, “רגע…”
  - irritation: “נו…”, “שנייה!”

#### 7.5.3 Avoid rapid style switching
- Keep a consistent style within a turn.
- Avoid toggling style every clause.

### 7.6 Quality gates to add (metrics and thresholds)
Start with these as defaults; calibrate with a labeled dev set.

#### 7.6.1 Per-turn gates (after TTS render)
- Duration: reject if > 30 s per turn (you should not have extremely long turns).
- Sustained-voicing: reject if any continuous voiced region > 6 s with very low spectral flux.
- Peak clipping: reject if >0.5% of samples are within 0.1 dB of full scale.

#### 7.6.2 Full-clip gates (after post-processing)
- Muffledness: LTAS high-band ratio too low (e.g., `E(4–7k)/E(0.3–4k) < -25 dB`).
- Click rate: >10 click candidates per minute above threshold.
- Speaker drift: embedding similarity per speaker below threshold (calibrate).

#### 7.6.3 Optional ML-based gates
- MOSNet/UTMOS (calibrate): flag bottom 5% as likely synthetic/glitched.
- DNSMOS for distortions (calibrate).

### 7.7 Code examples (practical snippets)

#### 7.7.1 Apply RIR convolution with torchaudio
```python
import torch
import torchaudio
import torchaudio.functional as F

wav, sr = torchaudio.load("dry.wav")
rir, rir_sr = torchaudio.load("rir.wav")
assert sr == 16000 and rir_sr == sr

# Normalize RIR energy
rir = rir / torch.sqrt(torch.sum(rir**2) + 1e-12)

# Convolve (FIR) - use fftconvolve for speed if needed
wet = F.fftconvolve(wav, rir)

torchaudio.save("wet.wav", wet, sr, encoding="PCM_S", bits_per_sample=16)
```

#### 7.7.2 Add “table reflection” (simple early reflection)
```python
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

x, sr = sf.read("wet.wav")
assert sr == 16000
if x.ndim > 1: x = x[:,0]

delay_ms = np.random.uniform(0.6, 1.2)
d = int(sr * delay_ms / 1000.0)
a = 10 ** (-np.random.uniform(6, 10) / 20.0)  # -6 to -10 dB

# LPF for reflection
b, c = butter(2, 5000/(sr/2), btype="low")
ref = lfilter(b, c, np.pad(x, (d,0))[:-d])

y = x + a * ref
y = np.clip(y, -1.0, 1.0)
sf.write("wet_table.wav", y, sr, subtype="PCM_16")
```

#### 7.7.3 Codec simulation with FFmpeg (Opus voip)
```bash
ffmpeg -i wet_table.wav -ar 48000 -ac 1 -c:a libopus -b:a 16k -application voip tmp.ogg
ffmpeg -i tmp.ogg -ar 16000 -ac 1 -c:a pcm_s16le out_final.wav
```

---

## Summary of the root causes in your current pipeline (and the fixes)
1) **Muffled/muddy**: you are stacking an aggressive low-pass and Wiener denoise; both reduce frication detail and spectral contrast.
   → Remove denoise; soften LPF or replace with mild EQ.
2) **Pitch drift / wrong anger pitch**: emotion control is likely implemented via pitch-level control instead of range/energy/timing.
   → Keep male mean F0 stable; increase range + intensity + timing changes; chunk synthesis and reset style.
3) **Robotic pacing**: overuse of slow rate and uniform pauses.
   → Use Hebrew speaking-rate anchors (~5–6 SPS) and vary pauses; add turn-taking cues.
4) **TTS artifacts**: long chunks, extreme SSML, style switching.
   → Chunk synthesis, safe parameter ranges, crossfade joins, and add quality-gate regeneration.
5) **No spatial realism**: missing room+device+codec.
   → Add RIR convolution + table reflection + ambience + codec simulation.

---

## References (author-year pointers used in report)
- Banse & Scherer (1996). Acoustic Profiles in Vocal Emotion Expression. https://www.columbia.edu/~rmk7/HC/HC_Readings/Scherer.pdf
- Schröder (2001). Acoustic correlates of emotion dimensions (Eurospeech). https://www.isca-archive.org/eurospeech_2001/schroder01_eurospeech.pdf
- Patman (2025). Acoustic methods for analysing breathy and whispery voices (systematic review). https://pure.york.ac.uk/portal/files/129149770/10.1515_phon-2025-0007.pdf
- Childers & Lee (1991). Vocal Quality Factors: Analysis, Synthesis, and Perception. https://linguistics.berkeley.edu/~kjohnson/LSA317/Childers%26Lee1991.pdf
- Silber-Varod (2016). Acoustic correlates of lexical stress in Israeli Hebrew. https://www.sciencedirect.com/science/article/abs/pii/S0095447016000048
- Amir (preliminary). Speaking Rate among Adult Hebrew Speakers (SPS/WPS anchors). https://www.researchgate.net/profile/Ofer-Amir/publication/303859712_Speaking_Rate_among_Adult_Hebrew_Speakers_A_Preliminary_Observation/links/5759045308ae414b8e3f6452/Speaking-Rate-among-Adult-Hebrew-Speakers-A-Preliminary-Observation.pdf
- Gelfer (2005). Speaking fundamental frequency ranges (male vs female). https://www.sciencedirect.com/science/article/abs/pii/S0892199704001729
- Google Cloud TTS Chirp3-HD docs. https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
- Google voice list (includes he-IL Chirp3-HD voices). https://docs.cloud.google.com/text-to-speech/docs/list-voices-and-types
- Azure Speech SSML prosody constraints. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
- Azure Speech language/voice support. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
- pyroomacoustics. https://github.com/LCAV/pyroomacoustics
- gpuRIR + paper. https://github.com/DavidDiazGuerra/gpuRIR and https://arxiv.org/abs/1810.11359
- OpenAIR IR library. https://www.openair.hosted.york.ac.uk/
- BUT ReverbDB. https://speech.fit.vut.cz/software/but-speech-fit-reverb-database
- OpenSLR SLR28 (RIR + noise). https://www.openslr.org/28/
- torchaudio augmentation tutorial (includes “noisy speech over phone”). https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
- PESQ/POLQA overview + licensing pointers. https://github.com/jonnor/machinehearing/blob/master/audio-quality/README.md and https://www.opticom.de/licensing/licensing.php
