# Engineering Naturalness: A Comprehensive Framework for Hebrew Synthetic Speech Generation and Realistic Post-Processing

The evolution of speech synthesis has transitioned from the mechanical concatenation of phonemes to the sophisticated generation of neural waveforms. However, achieving true "naturalness"—defined not just by intelligibility but by the subtle, organic imperfections of human communication—remains a significant challenge. This report delineates an exhaustive technical strategy for generating Hebrew-language synthetic audio that bypasses the "uncanny valley" of robotic synthesis. It addresses the specific psychoacoustic, linguistic, and signal-processing requirements necessary to produce datasets for robust audio classification models.

## Part 1: How Humans Perceive Speech Naturalness

The human auditory system is an incredibly sensitive instrument, evolved to detect minute variations in the acoustic signal that convey essential indexical information about a speaker’s biological state, emotional intent, and physical environment.<sup>1</sup> For synthetic speech to be perceived as "real," it must replicate the stochastic nature of the human vocal production system rather than adhering to the mathematically perfect, and thus robotic, outputs of early synthesis engines.

### Acoustic Features and the Perception of Realism

The perception of "real" versus "synthetic" speech hinges on the presence of micro-prosodic perturbations. Human vocal fold vibration is never perfectly periodic. This inherent instability is quantified through several key parameters:

| **Parameter** | **Acoustic Definition** | **Perceptual Role** |
| --- | --- | --- |
| **Jitter (local)** | Cycle-to-cycle variation in the fundamental frequency ().<sup>2</sup> | Provides "texture"; a lack of jitter sounds sterile or "plastic".<sup>4</sup> |
| --- | --- | --- |
| **Shimmer (local)** | Short-term variation in the peak-to-peak amplitude of the waveform.<sup>1</sup> | Reflects vocal stability; excessive shimmer indicates hoarseness, while zero shimmer indicates synthesis.<sup>3</sup> |
| --- | --- | --- |
| **Harmonics-to-Noise Ratio (HNR)** | The ratio of periodic energy to aperiodic noise.<sup>1</sup> | Low HNR correlates with breathiness or vocal fry; high HNR is perceived as "clean" but potentially artificial.<sup>4</sup> |
| --- | --- | --- |
| **Spectral Tilt** | The rate of energy decay across the frequency spectrum.<sup>7</sup> | Differentiates vocal effort; a shallow tilt (more high-frequency energy) is a primary marker of anger or shouting.<sup>10</sup> |
| --- | --- | --- |

Psychoacoustic research, such as the work by Klingholz and Martin (1985), suggests that jitter is particularly relevant for differentiating between functional voice states. In hyperfunctional states (e.g., intense anger or shouting), vocal fold tension increases, which paradoxically reduces jitter while potentially increasing spectral noise.<sup>6</sup> Synthetic engines often fail to model this inverse relationship, simply raising pitch without accounting for the corresponding changes in perturbation and tilt.

### The Role of Fundamental Frequency and Spectral Profiles

The fundamental frequency () is the most investigated prosodic feature and serves as a primary marker for arousal and valence.<sup>7</sup> During natural speech,  is not a static value but a dynamic contour that interacts with intensity (RMS energy). Higher synchrony between  and RMS energy is typically observed in high-activation emotions like anger and happiness.<sup>11</sup>

The "Alpha Ratio"—the energy balance between the 0–2000 Hz and 2000–5000 Hz regions—is another critical naturalness marker.<sup>7</sup> Research indicates that humans can maintain reliable perception of these voice quality measures even in the presence of background noise up to 47.7 dB(A), suggesting the brain uses these spectral ratios to "fingerprint" the speaker.<sup>2</sup> Synthetic speech that undergoes aggressive lowpass filtering (e.g., a 7.5 kHz cutoff at a 16 kHz sample rate) destroys the higher-order harmonics and the spectral "air" that characterize natural phonation.

### Emotional Manifestation in Real Speech

Emotion in speech is not merely a change in pitch; it is a holistic reconfiguration of the vocal tract's output driven by the autonomic nervous system.<sup>7</sup>

#### Acoustic Profiles of Discrete Emotions

Based on phonetics research (Banse & Scherer, 1996; Juslin & Laukka, 2003), the following ranges are observed in natural speech:

| **Emotion** | **F0 Mean** | **F0 Variability/Range** | **Speech Rate** | **Spectral Characteristics** |
| --- | --- | --- | --- | --- |
| **Anger** | Highly Elevated | Narrowed (due to high floor) | Fast | Flat spectral tilt; high energy in 2-5kHz.<sup>7</sup> |
| --- | --- | --- | --- | --- |
| **Fear** | Elevated | Increased/High | Fast | High jitter; breathy voice quality.<sup>11</sup> |
| --- | --- | --- | --- | --- |
| **Sadness** | Decreased | Small/Narrow | Slow | Steep spectral tilt; low HNR.<sup>12</sup> |
| --- | --- | --- | --- | --- |
| **Calm** | Baseline | Low/Uniform | Moderate | Balanced Alpha Ratio; stable HNR.<sup>12</sup> |
| --- | --- | --- | --- | --- |

A critical insight from Hebrew-specific research (Amir et al., 2003) is that while  mean is the strongest indicator of anger, the absolute  range actually exhibits a significant negative correlation with anger intensity.<sup>10</sup> This happens because the pitch minimum (the "floor") rises faster than the maximum, leading to a "constricted" but high-pitched delivery. Most TTS systems incorrectly expand the range when "angry" styles are selected, leading to an unnatural "musical" quality rather than the intended aggression.

### Hebrew-Specific Prosody Patterns

Hebrew presents unique challenges for synthetic naturalness due to its phonological structure and rhythmic class. Unlike English, which is stress-timed, modern Hebrew is often described as syllable-timed or intermediate, with a quantity-insensitive stress system.<sup>15</sup>

1.  **Stress and Duration:** The primary cue for Hebrew stress is duration. Stressed syllables are significantly longer, but this contrast is often eliminated in phrase-final positions.<sup>15</sup>
2.  **Iambic Bias:** The most frequent stress pattern is word-final (iambic), constituting approximately 75-85% of native nouns.<sup>15</sup>
3.  **Masoretic Accents (Teamim):** Historical Hebrew prosody was rooted in "accents" or "melodemes" that served as phrasing and intonation markers.<sup>18</sup> In modern speech, these survive as specific melodic contours that signal questions or emphasis. Synthetic engines that ignore these contours sound monotone and "foreign."

### Recorded Phone Audio vs. Studio Quality

Synthetic speech sounds artificial because it is "too perfect"—it lacks the acoustic signature of the recording chain and environment.

- **Frequency Response:** Mobile phones typically emphasize frequencies around 30 Hz and between 85-140 Hz.<sup>20</sup> Conversely, telephony codecs (AMR-NB) aggressively cut frequencies below 300 Hz and above 3400 Hz.<sup>21</sup>
- **Microphone Directivity:** Unlike studio cardioid microphones, smartphones use omnidirectional primary mics that are heavily influenced by the phone's body shape, particularly at high frequencies.<sup>22</sup>
- **Room Interaction:** Real phone audio captures early reflections (the first 50ms) and late reverberation (RT60), which provide spatial context. A "phone on a table" produces a distinct comb-filtering effect due to the reflection off the hard table surface.<sup>23</sup>

## Part 2: State of the Art in Neural TTS Naturalness

Modern neural TTS systems like Azure Neural TTS and Google Chirp 3 HD have reached high levels of intelligibility, but their implementation of Hebrew often relies on generic models that struggle with the language's specific phonetic and emotional demands.

### Azure Neural TTS: Capabilities and Hebrew Voices

Azure provides two primary Hebrew neural voices: he-IL-AvriNeural (Male) and he-IL-HilaNeural (Female). These are "Neural" voices, meaning they use a neural vocoder (like HiFi-GAN or WaveNet) to generate waveforms directly from spectrographic representations.<sup>25</sup>

#### SSML Implementation and Under-the-Hood Mechanics

The Speech Synthesis Markup Language (SSML) serves as the control layer for these models.<sup>26</sup>

- **Prosody Tag:** Adjusts the rate, pitch, and volume. These changes are applied as bias vectors in the latent space of the neural model.<sup>28</sup>
- **Express-As Style:** Allows for predefined styles like cheerful, angry, or whispering. However, for Hebrew, many of these styles are either unavailable or "blended," leading to identity shifts where the voice's timbre changes entirely when a style is applied.<sup>13</sup>
- **Contour Control:** Azure supports &lt;prosody contour="..."&gt;, which allows developers to set specific  targets at different percentages of a sentence's duration. This is the most powerful tool for preventing "flat" speech, yet it is often underutilized.<sup>28</sup>

#### Known Issues with Azure Neural TTS

1.  **Identity Shifts:** Because Azure does not support version pinning for standard neural voices, the backend model can be updated without notice, altering the timbre, pitch, and perceived gender of the voice (e.g., the reported issues with AvaMultilingualNeural).<sup>30</sup>
2.  **Pacing and Artifacts:** Short transition sentences are often spoken too fast if not guided by &lt;prosody rate&gt;. Additionally, Azure treats all-caps words (e.g., "MIN") as initialisms ("M-I-N") by default, which contributes to the "robotic" perception.<sup>28</sup>
3.  **Boundary Clicks:** Abrupt transitions between SSML blocks can cause DC-offset jumps, manifesting as clicks.<sup>31</sup>

### Google Cloud TTS Chirp 3 HD

Google's Chirp 3 HD is a part of the Universal Speech Model (USM) ecosystem. It is designed for massive language breadth and is generally superior at handling the "long-tail" of language nuances compared to Azure's Western-centric models.<sup>33</sup>

- **Capabilities:** Chirp 3 HD supports "Instant Custom Voice," allowing users to create unique models from small audio samples.<sup>34</sup>
- **Limitations:** The primary drawback of Chirp 3 HD is latency. For long-form audio (e.g., 5-minute clips), latency can reach up to 3.5 seconds, and catastrophic speed drops have been reported.<sup>35</sup>
- **Hebrew Failure Modes:** Users report that Chirp models frequently fail on single-syllable Hebrew words, especially monosyllabic ones, which it may skip or mispronounce about 10% of the time.<sup>37</sup>

### Comparison of Modern Engines

| **Engine** | **Pros** | **Cons** | **Ideal Use Case** |
| --- | --- | --- | --- |
| **Azure Neural** | Stable, professional, deep SSML integration.<sup>33</sup> | Flat newsreader prosody; identity shifts.<sup>30</sup> | Corporate/Informational. |
| --- | --- | --- | --- |
| **Google Chirp 3 HD** | Exceptional language accuracy; custom voice support.<sup>33</sup> | High latency; single-syllable failures.<sup>35</sup> | High-accuracy multilingual. |
| --- | --- | --- | --- |
| **ElevenLabs** | Industry-leading emotional range and cinematic performance.<sup>33</sup> | High cost ($0.05/min); credit-based opacity.<sup>36</sup> | Storytelling/High-emotion. |
| --- | --- | --- | --- |
| **OpenAI TTS-1** | Fast, simple, low latency.<sup>33</sup> | Limited prosody control; 45.8% prosody accuracy.<sup>36</sup> | Real-time assistants. |
| --- | --- | --- | --- |

### Professional SSML Hacks for Audiobook Naturalness

Professional producers do not rely on default styles. Instead, they use "micro-SSML" to simulate biological breathing and cognitive pauses.

- **The "Breath" Hack:** Inserting &lt;break time="150ms"/&gt; after commas and &lt;break time="400ms"/&gt; between paragraphs simulates the natural inhalation process.<sup>42</sup>
- **Punctuation Weighting:** Many neural engines interpret multiple exclamation marks (!!!!!) as a cue to increase arousal and volume beyond what a single tag can do.<sup>44</sup>
- **Substitutions for Flow:** Using the &lt;sub&gt; tag to replace acronyms with their phonetic counterparts (e.g., &lt;sub alias="min"&gt;MIN&lt;/sub&gt;) ensures consistent pacing.<sup>28</sup>
- **Onomatopoeia Stretching:** Stretching vowels in text (e.g., "ahhh...", "huuuuh...") can force the neural vocoder to produce more expressive, non-linguistic vocalizations.<sup>44</sup>

## Part 3: Post-Processing for Realism

The "studio-clean" output of TTS is a major contributor to its perceived artificiality. Realistic recordings are characterized by room acoustics, device coloration, and ambient noise.

### Room Acoustics Simulation

Simulating the acoustic environment requires modeling both early reflections and late reverberation.

1.  **Pyroomacoustics:** This Python library uses the Image-Source Method (ISM) for shoebox rooms and ray-tracing for non-convex geometries.<sup>23</sup> It is ideal for simulating small apartments or offices.
    - **Parameters:** For a typical room (), the absorption coefficients () should be frequency-dependent.
    - **Example Code:**

Python
import pyroomacoustics as pra
room = pra.Shoebox(\[4, 3, 2.5\], fs=16000,
materials=pra.Material(0.15), max_order=10)
room.add_source(\[1, 1.5, 1.2\], signal=audio_data)
room.add_microphone()
room.simulate()

1.  **gpuRIR:** For large-scale dataset generation, gpuRIR provides massive speedups by offloading the convolution of Room Impulse Responses (RIRs) to the GPU.<sup>47</sup>

### Device and Channel Simulation

The acoustic profile of a mobile phone on a table involves two main components: the physical transducer response and the digital codec compression.

- **Microphone Frequency Response:** To simulate a budget phone mic, apply a high-shelf filter to boost high-frequency noise and a low-cut filter at 100 Hz to remove "bass" that small diaphragms cannot capture.<sup>22</sup>
- **Codec Simulation (AMR/Opus):** Telephony audio is dominated by the Adaptive Multi-Rate (AMR) codec.
    - **AMR-NB:** Sampling at 8 kHz, bitrate 4.75–12.2 kb/s. This creates the classic "muffled" phone sound by cutting everything above 3.4 kHz.<sup>21</sup>
    - **Artifacts:** AMR introduces "white islands" (gaps in the spectrogram) that confuse audio classification models if they are only trained on clean audio.<sup>21</sup>
- **Denoising Strategy:** Avoid Wiener denoising on clean TTS. Instead, use a "Noise Gate" to ensure silence is absolute between turns, but maintain a low-level "Room Tone" (pink noise at -45 dBFS) to ground the voice.<sup>32</sup>

### Background Noise and SNR

The Signal-to-Noise Ratio (SNR) determines the perceived distance and clarity of the speaker.

| **Condition** | **Typical SNR Range (dB)** | **Acoustic Profile** |
| --- | --- | --- |
| **Phone on Table** | 25 – 35 | Distant, includes room reverb and table reflections.<sup>20</sup> |
| --- | --- | --- |
| **Phone in Pocket** | 5 – 15 | Heavy low-frequency "rubbing" noise; severe high-frequency attenuation.<sup>51</sup> |
| --- | --- | --- |
| **Studio/Clean** | \> 60 | No context; unnatural for classification models.<sup>2</sup> |
| --- | --- | --- |

## Part 4: Pitch and Prosody Engineering

The failure of synthetic speech to sound natural often stems from a misunderstanding of how pitch and rate scale with emotion and gender in the Hebrew language.

### Measured Pitch Ranges for Hebrew Speakers

Gender in Hebrew speech is distinguished by both  and formant frequencies (vocal tract length).<sup>52</sup>

| **Gender Group** | **Mean F0 (Hz)** | **F0 Range (SD)** | **Typical F1 (Hz)** |
| --- | --- | --- | --- |
| **Cisgender Male** | 100 – 146 | ~15 Hz | 400 – 600.<sup>52</sup> |
| --- | --- | --- | --- |
| **Cisgender Female** | 188 – 221 | ~30 Hz | 500 – 800.<sup>52</sup> |
| --- | --- | --- | --- |
| **Boundary/Neutral** | 155 – 180 | \-  | \-.<sup>52</sup> |
| --- | --- | --- | --- |

When engineering "Angry" speech for a male speaker, the pitch should not merely rise; it should stabilize at a higher "floor." In authentic anger, the  minimum rises, while the maximum rise is weak and insignificant.<sup>10</sup> This leads to a higher mean  but a lower overall  range (SD).

### Hebrew Emotional Prosody (T-RES Parameters)

The Test for Rating Emotions in Speech (T-RES) provides the "gold standard" for Hebrew emotional acoustics.<sup>14</sup>

| **Emotion** | **Mean F0 (Norm. Hz)** | **F0 Range (Norm. Hz)** | **Speech Rate (Syll/Sec)** |
| --- | --- | --- | --- |
| **Anger** | 2.47 | 3.21 | 4.95.<sup>14</sup> |
| --- | --- | --- | --- |
| **Happiness** | 2.11 | 2.41 | 4.14.<sup>14</sup> |
| --- | --- | --- | --- |
| **Sadness** | 1.33 | 2.85 | 3.77.<sup>14</sup> |
| --- | --- | --- | --- |
| **Neutral** | 0.53 | 1.53 | 4.41.<sup>14</sup> |
| --- | --- | --- | --- |

Note: Normalized Hz is calculated as .<sup>14</sup>

### Preventing Pitch Drift in Multi-Turn TTS

Pitch drift—where a speaker’s pitch steadily climbs across a 5-minute clip—is a common failure in neural TTS state management.

- **Technique:** Use the &lt;prosody pitch="default"&gt; tag at the start of every new speaker turn to reset the engine's internal state.
- **Contour Mapping:** For long utterances, use a pitch contour that mimics the "declination" effect (natural pitch drop towards the end of a sentence).
    - **Math:** , where  is the declination constant.

### Micro-Prosody: Jitter and Shimmer in Post-Processing

While neural TTS generates clean waves, jitter and shimmer can be reintroduced using the Parselmouth (Praat) library.<sup>55</sup>

- **Jitter Injection:** Modify the timing of glottal pulses in a PointProcess object. Adding a random delta (jitter ) breaks the perfect periodicity of the synthetic wave.<sup>8</sup>
- **Shimmer Injection:** Multiply the amplitude of individual glottal cycles by a random scaling factor (shimmer ).<sup>5</sup>

## Part 5: Eliminating TTS Artifacts

Automating the cleaning of a synthetic dataset is essential to ensure that training models do not learn the "noise" of the generator itself.

### Common Azure Artifacts and Causes

1.  **Sustained Vowels:** Often caused by a failure in the neural vocoder's attention mechanism on long words or all-caps text.
    - **Detection:** Use a "Spectral Anomaly Detector" to flag frames where the energy remains constant for .<sup>28</sup>
2.  **Boundary Clicks:** Result from non-zero-crossing audio cuts.
    - **Fix:** Apply a 10ms crossfade between every sentence or turn.
3.  **Voice Identity Shifts:** Occur when switching express-as styles mid-dialogue.
    - **Fix:** Generate each emotion as a separate audio file and join them using consistent speaker models, or avoid express-as entirely in favor of &lt;prosody&gt; tags.<sup>30</sup>

### Filtering Broken Output

Use a combination of Voice Activity Detection (VAD) and RMS energy levels to automatically filter clips.

- **Logic:** A frame counts as valid "speech" only if BOTH the VAD probability and the volume are above thresholds.<sup>57</sup>
- **Silence Thresholds:** Clips with silence ratios  or pauses  should be discarded or re-sliced.<sup>32</sup>

## Part 6: Existing Projects and Tools

Leveraging open-source libraries and established datasets is critical for benchmarking the quality of the generated audio.

### Audio Augmentation Libraries

| **Library** | **Best For** | **Features** |
| --- | --- | --- |
| **Audiomentations** | CPU-based augmentation.<sup>59</sup> | Pitch shift, time stretch, background noise.<sup>60</sup> |
| --- | --- | --- |
| **Torch-audiomentations** | GPU-accelerated pipelines.<sup>62</sup> | High-speed batch processing for deep learning. |
| --- | --- | --- |
| **Pedalboard** | High-quality VST effects.<sup>59</sup> | Reverb, compression, and high-quality EQ.<sup>43</sup> |
| --- | --- | --- |
| **Parselmouth** | Acoustic analysis and manipulation.<sup>8</sup> | Precise control over jitter, shimmer, and HNR.<sup>55</sup> |
| --- | --- | --- |

### Room Impulse Response (RIR) Databases

Using real RIRs is more effective than artificial ones for training robust models.<sup>63</sup>

- **dEchorate:** Annotated multichannel RIRs for small cuboid rooms.<sup>65</sup>
- **BUT ReverbDB:** Real RIRs from 8 different rooms (small, medium, large).<sup>63</sup>
- **EchoThief:** Diverse RIRs from non-traditional spaces.<sup>55</sup>
- **MYRiAD:** Multi-array RIRs from two recording spaces.<sup>67</sup>

### Speech Quality Assessment Tools

Instead of manual listening tests, use non-intrusive MOS (Mean Opinion Score) predictors:

- **DNSMOS:** Measures speech quality in the presence of noise (scale 1-5).<sup>68</sup>
- **UTMOS:** Specifically designed for calculating MOS for synthetic voice samples.<sup>70</sup>
- **NISQA:** Evaluates quality, noisiness, and coloration.<sup>71</sup>

## Part 7: Recommended Architecture Changes

To address the quality issues (muffled sound, pitch drift, robotic pacing), the following concrete modifications are recommended for the pipeline.

### 1\. Pre-processing and SSML Strategy

- **Remove All-Caps:** Lowercase all text to prevent Azure from spelling out words like acronyms.<sup>28</sup>
- **Fix Pacing:** For low-intensity or calm speech, manually set &lt;prosody rate="95%"&gt;. For high-arousal (anger), use &lt;prosody rate="115%"&gt;.<sup>14</sup>
- **Pitch Floor Control:** For angry male speakers, use &lt;prosody pitch="+5%" contour="(0%, 0%) (100%, -2%)"&gt;. The rise in pitch should be accompanied by a _reduction_ in pitch variability.<sup>10</sup>
- **Style Management:** Abandon mstts:express-as="angry" if it causes identity shifts. Instead, use a custom prosody block:
    XML
    &lt;prosody pitch="+2st" rate="+10%" volume="loud"&gt;
    &lt;emphasis level="strong"&gt;זה לא יכול להיות!&lt;/emphasis&gt;
    &lt;/prosody&gt;


### 2\. Post-Processing Pipeline (The "Realism" Layer)

- **REMOVE Wiener Denoising:** This is the primary cause of muddy sound on clean TTS.
- **Modify Filtering:** Replace the 7.5 kHz lowpass with a high-pass filter at 100 Hz (to remove "DC" and infrasonic synthetic noise) and a gentle wide-band codec simulation.<sup>21</sup>
- **Add "Phone-on-Table" Convolution:** Use pyroomacoustics to convolve the audio with a RIR that includes a strong early reflection from a surface 2cm below the microphone.<sup>23</sup>
- **Codec Simulation:** Apply AMR-NB encoding at 12.2 kbps to introduce realistic quantization artifacts and band-limiting.<sup>21</sup>

### 3\. Quality Gates

- **Artifact Detection:** Run every clip through UTMOS. Reject clips with a score .<sup>70</sup>
- **Silence Gate:** Use audio-slicer with a threshold of -40 dB. If silence blocks  are detected within a speaker turn, flag for regeneration.<sup>32</sup>
- **Spectral Continuity:** Measure the "Alpha Ratio".<sup>7</sup> If the ratio deviates  from the average for that speaker/emotion pair, it indicates a likely synthesis glitch.

### 4\. Implementation Priority Ordering

1.  **High Gain / Low Effort:** Remove Wiener denoising; lower-case all input text; add 10ms crossfades at boundaries.
2.  **High Gain / Medium Effort:** Implement AMR-NB codec simulation; use T-RES prosody values instead of Azure styles.
3.  **Medium Gain / High Effort:** Integrate pyroomacoustics for physical spatialization; use Parselmouth to reintroduce jitter and shimmer.

By implementing these structural and algorithmic changes, the pipeline will shift from producing "robotic voices in a vacuum" to a dataset of "human-like communication in realistic environments," significantly improving the performance of downstream audio classification models.

#### Works cited

1.  A COMPREHENSIVE REVIEW OF JITTER, SHIMMER, AND HNR: LINGUISTIC AND PARALINGUISTIC APPLICATIONS - The Repository at St. Cloud State, accessed April 30, 2026, https://repository.stcloudstate.edu/cgi/viewcontent.cgi?params=/context/stcloud_ling/article/1155/&path_info=1_Koffi2025ComprehensiveReviewOfJitterShimmerHNR.pdf
2.  Variation of the acoustic parameters: f0, jitter, shimmer and alpha ratio in relation with different background noise levels | Request PDF - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/364395991_Variation_of_the_acoustic_parameters_f0_jitter_shimmer_and_alpha_ratio_in_relation_with_different_background_noise_levels
3.  (PDF) Vocal Acoustic Analysis – Jitter, Shimmer and HNR Parameters - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/273822486_Vocal_Acoustic_Analysis_-_Jitter_Shimmer_and_HNR_Parameters
4.  Measuring negative emotions and stress through acoustic correlates in speech: A systematic review | PLOS One - Research journals, accessed April 30, 2026, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0328833
5.  Analyzing Vocal Features for Pathology — SpeechBrain 0.5.0 documentation, accessed April 30, 2026, https://speechbrain.readthedocs.io/en/latest/tutorials/preprocessing/voice-analysis.html
6.  Quantitative spectral evaluation of shimmer and jitter - PubMed, accessed April 30, 2026, https://pubmed.ncbi.nlm.nih.gov/4010246/
7.  Measuring negative emotions and stress through acoustic correlates in speech: A systematic review - PMC, accessed April 30, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12289014/
8.  Speech Analysis using Parselmouth | by Satish Kumar Andey - Medium, accessed April 30, 2026, https://satishkumarandey.medium.com/speech-analysis-using-parselmouth-7bb1a2760cc9
9.  Classifying Subject Ratings of Emotional Speech Using Acoustic Features. - ISCA Archive, accessed April 30, 2026, https://www.isca-archive.org/eurospeech_2003/liscombe03_eurospeech.pdf
10. Characteristics of authentic anger in Hebrew speech - ISCA Archive, accessed April 30, 2026, https://www.isca-archive.org/eurospeech_2003/amir03_eurospeech.pdf
11. ASA 148th Meeting Lay Language Papers -Study of acoustic correlates associated with emotional speech, accessed April 30, 2026, https://acoustics.org/pressroom/httpdocs/148th/yildirim.html
12. Acoustic Patterns of Emotions - Vox Institute, accessed April 30, 2026, https://www.vox-institute.ch/publications/Zei-Acoustic-patternsCHAP23-2002.pdf
13. Voice and sound with Speech Synthesis Markup Language (SSML) - Microsoft Learn, accessed April 30, 2026, https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
14. A Cross-Linguistic Validation of the Test for Rating Emotions in ..., accessed April 30, 2026, https://pubs.asha.org/doi/10.1044/2021_JSLHR-21-00205
15. Modern Hebrew stress | Outi Bat-El, accessed April 30, 2026, https://www.outibatel.com/wp-content/uploads/2010/12/Bat-El-Cohen-and-Silber-Varod-2019-Hebrew-stress.pdf
16. Rhythm Perception in Speakers of Arabic, German and Hebrew - PMC - NIH, accessed April 30, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC11700906/
17. Prosodic patterns in Hebrew child-directed speech - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/23471004_Prosodic_patterns_in_Hebrew_child-directed_speech
18. Phonology of Masoretic Hebrew II Accents As Prosody - Univerzita Karlova, accessed April 30, 2026, https://dspace.cuni.cz/bitstream/handle/20.500.11956/94176/140059603.pdf
19. THE PROSODIC BASIS OF THE TIBERIAN HEBREW SYSTEM OF ACCENTS BEZALEL ELAN DRESHER - Cambridge University Press & Assessment, accessed April 30, 2026, https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4B9357970E8D5222D7F3865F46640EC4/S0097850794048095a.pdf/the-prosodic-basis-of-the-tiberian-hebrew-system-of-accents.pdf
20. Smartphone recordings are comparable to “gold standard” recordings for acoustic measurements of voice - PMC, accessed April 30, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC10545813/
21. Impact of the GSM AMR Codec on Automatic Vowel Formant Measurement in Praat and VoiceSauce - Tomáš Bořil, accessed April 30, 2026, https://tomasboril.cz/publications/kaiser_boril2018_impact_gsm_amr_codec_vowel_formant.pdf
22. Acoustical Measurements with Smartphones: Possibilities and Limitations, accessed April 30, 2026, https://acousticstoday.org/wp-content/uploads/2017/06/2-faber.pdf
23. Room Simulation — Pyroomacoustics 0.10.0 documentation - Read the Docs, accessed April 30, 2026, https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
24. The Acoustical design of Mobile phones, accessed April 30, 2026, https://www.acoustics.asn.au/conference_proceedings/ICA2010/cdrom-ICA2010/papers/p489.pdf
25. Azure-Speech-Text-to-speech - AI Model Catalog | Microsoft Foundry Models, accessed April 30, 2026, https://ai.azure.com/catalog/models/Azure-Speech-Text-to-speech
26. Speech Synthesis Markup Language (SSML) | Cloud Text-to-Speech | Google Cloud Documentation, accessed April 30, 2026, https://docs.cloud.google.com/text-to-speech/docs/ssml
27. Speech Synthesis Markup Language (SSML) document structure and events - Speech service - Foundry Tools | Microsoft Learn, accessed April 30, 2026, https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure
28. Pronunciation issue when generating audio from SSML script using Azure Speech Service, accessed April 30, 2026, https://learn.microsoft.com/en-us/answers/questions/5729867/pronunciation-issue-when-generating-audio-from-ssm
29. Expressive TTS with SSML: Advanced Techniques, Testing, and API Integration - DupDub, accessed April 30, 2026, https://www.dupdub.com/blog/expressive-tts-ssml-guide
30. Change in TTS voice - Microsoft Q&A, accessed April 30, 2026, https://learn.microsoft.com/en-us/answers/questions/5771581/change-in-tts-voice
31. Azure TTS low voice quality in AP - Support Questions - ATOMI Community, accessed April 30, 2026, https://talk.atomisystems.com/t/azure-tts-low-voice-quality-in-ap/6440
32. openvpi/audio-slicer: Python script that slices audio with silence detection - GitHub, accessed April 30, 2026, https://github.com/openvpi/audio-slicer
33. Azure Cognitive Speech Alternatives in 2026: The "Safe Choice" vs. The Specialists, accessed April 30, 2026, https://dasha.ai/tips/azure-cognitive-speech-alternatives
34. Compare Azure AI Speech vs. Chirp 3 in 2026 - Slashdot, accessed April 30, 2026, https://slashdot.org/software/comparison/Azure-AI-Speech-vs-Chirp-3/
35. TTS Latency Benchmark 2025: Google vs. Microsoft Voices for Phonebots - BOTfriends, accessed April 30, 2026, https://botfriends.de/en/blog/tts-latency-benchmark-2025-google-vs-microsoft-voices-fuer-phonebots/
36. 8 Best Text-to-Speech APIs for Developers (2026 Comparison) - Inworld AI, accessed April 30, 2026, https://inworld.ai/resources/best-text-to-speech-apis
37. For those looking for "100% realistic TTS", the new Google Chirp HD voices are INSANE : r/learnthai - Reddit, accessed April 30, 2026, https://www.reddit.com/r/learnthai/comments/1jv03xi/for_those_looking_for_100_realistic_tts_the_new/
38. Best TTS APIs in 2026: ElevenLabs, Google, AWS & 9 More Compared - Speechmatics, accessed April 30, 2026, https://www.speechmatics.com/company/articles-and-news/best-tts-apis-in-2025-top-12-text-to-speech-services-for-developers
39. Real production comparison: ElevenLabs vs PlayHT vs Azure TTS vs Cartesia for phone-quality voice AI : r/artificial - Reddit, accessed April 30, 2026, https://www.reddit.com/r/artificial/comments/1ra81v9/real_production_comparison_elevenlabs_vs_playht/
40. I Compared 4 TTS/STT Engines for My AI Homelab. Here's What Actually Matters. - Medium, accessed April 30, 2026, https://medium.com/@linglijunmail/i-compared-4-tts-stt-engines-for-my-ai-homelab-heres-what-actually-matters-ba24247c0ede
41. Best Voice AI Platforms: Enterprise Comparison 2026 - Deepgram, accessed April 30, 2026, https://deepgram.com/learn/best-voice-ai-platforms-enterprise-comparison
42. SSML Guide: Getting Natural-Sounding AI Speech | Kaizen Apps Blog, accessed April 30, 2026, https://kaizen-apps.com/blog/speech-studio/ssml-guide.html
43. How to Make Text-to-Speech Sound Less Robotic & More Humanlike - Voice.ai, accessed April 30, 2026, https://voice.ai/hub/tts/how-to-make-text-to-speech-sound-less-robotic/
44. How to Make Text-to-Speech Moan & Improve Vocal Expression - Voice.ai, accessed April 30, 2026, https://voice.ai/hub/tts/how-to-make-text-to-speech-moan/
45. Making TTS with vivid emotion - Supertone, accessed April 30, 2026, https://www.supertone.ai/en/work/lively-tts-voice-guide-eng
46. Comparing Open Source Room Acoustics Simulation Tools: Performance and Usability Insights, accessed April 30, 2026, https://dael.euracoustics.org/confs/fa2025/data/articles/000229.pdf
47. Differences with other libraries · Issue #126 · LCAV/pyroomacoustics - GitHub, accessed April 30, 2026, https://github.com/LCAV/pyroomacoustics/issues/126
48. arXiv:1810.11359v4 \[eess.AS\] 9 Oct 2020, accessed April 30, 2026, https://arxiv.org/pdf/1810.11359
49. GSM Adaptive Multi-Rate (AMR) Codec - VOCAL Technologies, accessed April 30, 2026, https://vocal.com/speech-coders/gsm-amr/
50. emmaricomartin/speech-analysis-module: SAM is a Python library for sound/audio processing in experimental psychology - GitHub, accessed April 30, 2026, https://github.com/emmaricomartin/speech-analysis-module
51. (PDF) Sound-based proximity detection with mobile phones - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/262255704_Sound-based_proximity_detection_with_mobile_phones
52. Examining the voice of Israeli transgender women: Acoustic measures, voice femininity and voice-related quality-of-life - PMC, accessed April 30, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC8118229/
53. Examining the voice of Israeli transgender women: Acoustic measures, voice femininity and voice-related quality-of-life - Tel Aviv University, accessed April 30, 2026, https://cris.tau.ac.il/en/publications/examining-the-voice-of-israeli-transgender-women-acoustic-measure/
54. Fundamental frequency (F0) values for the 20 male and 20 female... - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/figure/Fundamental-frequency-F0-values-for-the-20-male-and-20-female-speakers-used-in-this_fig1_26702917
55. Parselmouth for bioacoustics: automated acoustic ... - MPG.PuRe, accessed April 30, 2026, https://pure.mpg.de/rest/items/item_3548150_3/component/file_3583821/content?download=true
56. voicelab - PyPI, accessed April 30, 2026, https://pypi.org/project/voicelab/
57. Tackling Turn Detection in Voice AI: Overcoming Noise and Interruption Challenges - Notch, accessed April 30, 2026, https://www.notch.cx/post/turn-detection-in-voice-ai
58. I built a small open-source speech fluency analyzer in Python - Reddit, accessed April 30, 2026, https://www.reddit.com/r/Python/comments/1rqotu3/i_built_a_small_opensource_speech_fluency/
59. Yuan-ManX/audio-development-tools: Audio Development Tools (ADT) is a project for advancing sound, speech, and music technologies, featuring components for machine learning, sound synthesis, speech and music generation, signal processing, game audio, digital audio workstations (DAWs), and more. · GitHub, accessed April 30, 2026, https://github.com/Yuan-ManX/audio-development-tools
60. A Survey of Data Augmentation for Audio Classification - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/375761305_A_Survey_of_Data_Augmentation_for_Audio_Classification
61. 10x Your AI Dataset FREE in 20 Min: Augmentation (2026) | Local AI Master, accessed April 30, 2026, https://localaimaster.com/tutorials/datasets/data-augmentation
62. Data Augmentation: What It Is & Why AI Models Need It - Kanerika, accessed April 30, 2026, https://kanerika.com/blogs/data-augmentation/
63. \[1811.06795\] Building and Evaluation of a Real Room Impulse Response Dataset - arXiv, accessed April 30, 2026, https://arxiv.org/abs/1811.06795
64. Building and Evaluation of a Real Room Impulse Response Dataset - Faculty of Information Technology, accessed April 30, 2026, https://www.fit.vut.cz/research/group/speech/public/publi/2019/szoke_IEEEjournal_2019_final_published_08717722-1.pdf
65. (PDF) dEchorate: a calibrated room impulse response dataset for echo-aware signal processing - ResearchGate, accessed April 30, 2026, https://www.researchgate.net/publication/356489365_dEchorate_a_calibrated_room_impulse_response_dataset_for_echo-aware_signal_processing
66. \[2104.13168\] dEchorate: a Calibrated Room Impulse Response Database for Echo-aware Signal Processing - arXiv, accessed April 30, 2026, https://arxiv.org/abs/2104.13168
67. MYRiAD: a multi-array room acoustic database - PMC, accessed April 30, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC10133077/
68. Deep Noise Suppression Mean Opinion Score (DNSMOS) - Lightning AI, accessed April 30, 2026, https://lightning.ai/docs/torchmetrics/stable/audio/deep_noise_suppression_mean_opinion_score.html
69. speechmos - PyPI, accessed April 30, 2026, https://pypi.org/project/speechmos/
70. GitHub - fakerybakery/utmos: A toolkit to calculate speech audio quality. Not affiliated with the original authors, accessed April 30, 2026, https://github.com/fakerybakery/utmos
71. Speech Quality Assessment for Enhanced Speech - Emergent Mind, accessed April 30, 2026, https://www.emergentmind.com/topics/speech-quality-assessment-for-enhanced-speech
