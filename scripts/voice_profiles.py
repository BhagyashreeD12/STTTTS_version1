"""
voice_profiles.py — Engine-agnostic voice profile registry.

Each profile describes a TTS voice in a provider-independent way.
The active profile is selected via the VOICE_PROFILE environment variable
(default: "kokoro_heart").

Usage
-----
    from voice_profiles import get_active_voice_profile, list_voice_profiles

    profile = get_active_voice_profile()
    print(profile.display_name)   # "Kokoro — Heart (af_heart)"
    print(profile.speed)          # 1.1
    print(profile.voice_id)       # "af_heart"

Selecting a different voice
---------------------------
    Set VOICE_PROFILE in your .env:

        VOICE_PROFILE=kokoro_michael

    Or at runtime:
        import os; os.environ["VOICE_PROFILE"] = "piper_amy"
        from voice_profiles import get_active_voice_profile   # re-reads env each call

Extending with new profiles
----------------------------
    Add a VoiceProfile(...) call to _register() at the bottom of each section.
    Required fields: id, provider, display_name.
    Provide whichever optional fields the target engine needs.

Supported providers (wiring required per engine in voice_assistant.py)
-----------------------------------------------------------------------
    kokoro      — model-file + voice_id  (currently wired)
    piper       — model_path + config_path
    chatterbox  — reference_audio
    xtts        — model_path + reference_audio
    qwen3       — voice_id / extra kwargs
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceProfile:
    """Immutable descriptor for a single TTS voice configuration.

    Fields
    ------
    id              Unique key used in VOICE_PROFILE env var and registry lookup.
    provider        Engine family: "kokoro" | "piper" | "chatterbox" | "xtts" | "qwen3".
    display_name    Human-readable label shown in logs / UI.
    description     One-line description of the voice character.
    voice_id        Provider-internal speaker name / id (e.g. "af_heart" for Kokoro).
    model_path      Path to .onnx / .pt / model folder (local model-file engines).
    config_path     Path to voice config JSON (Piper requires this alongside the .onnx).
    reference_audio Path to WAV used for voice conditioning (Chatterbox, XTTS-style).
    speed           TTS playback speed multiplier (1.0 = normal).
    language        BCP-47 language tag recognised by the engine (e.g. "en-us").
    style           Optional prosody/style hint — engine-specific, ignored if unsupported.
    extra           Catch-all dict for any additional provider-specific keyword arguments.
    """
    id:              str
    provider:        str
    display_name:    str
    description:     str           = ""
    voice_id:        Optional[str] = None
    model_path:      Optional[str] = None
    config_path:     Optional[str] = None
    reference_audio: Optional[str] = None
    speed:           float         = 1.0
    language:        str           = "en-us"
    style:           Optional[str] = None
    extra:           dict          = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_PROFILES: dict[str, "VoiceProfile"] = {}


def _register(*profiles: VoiceProfile) -> None:
    """Add one or more profiles to the global registry."""
    for p in profiles:
        _PROFILES[p.id] = p


# ===========================================================================
# ── Kokoro ONNX profiles
#    Requires: voice_id (speaker name from the voices-v1.0.bin bundle)
#    Engine:   kokoro_onnx  (currently active default in voice_assistant.py)
# ===========================================================================
_register(
    VoiceProfile(
        id           = "kokoro_heart",
        provider     = "kokoro",
        display_name = "Kokoro — Heart (af_heart)",
        description  = "Warm American female voice, natural cadence. Good for intake calls.",
        voice_id     = "af_heart",
        speed        = 1.1,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "kokoro_bella",
        provider     = "kokoro",
        display_name = "Kokoro — Bella (af_bella)",
        description  = "Bright, energetic American female voice.",
        voice_id     = "af_bella",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "kokoro_sky",
        provider     = "kokoro",
        display_name = "Kokoro — Sky (af_sky)",
        description  = "Calm, professional American female voice.",
        voice_id     = "af_sky",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "kokoro_michael",
        provider     = "kokoro",
        display_name = "Kokoro — Michael (am_michael)",
        description  = "Deep, authoritative American male voice.",
        voice_id     = "am_michael",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "kokoro_adam",
        provider     = "kokoro",
        display_name = "Kokoro — Adam (am_adam)",
        description  = "Natural American male voice.",
        voice_id     = "am_adam",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "kokoro_emma",
        provider     = "kokoro",
        display_name = "Kokoro — Emma (bf_emma)",
        description  = "Warm British female voice.",
        voice_id     = "bf_emma",
        speed        = 1.0,
        language     = "en-gb",
    ),
)


# ===========================================================================
# ── Piper TTS profiles
#    Requires: model_path (.onnx) + config_path (.onnx.json)
#    Install:  pip install piper-tts
#    Models:   https://huggingface.co/rhasspy/piper-voices
#    Notes:    Set model_path / config_path to absolute paths or paths relative
#              to the project root once you download the files.
# ===========================================================================
_register(
    VoiceProfile(
        id           = "piper_amy",
        provider     = "piper",
        display_name = "Piper — Amy (en_US/amy/medium)",
        description  = "Clear American female voice. Lightweight & fully offline.",
        voice_id     = "en_US-amy-medium",
        model_path   = "models/piper/en_US-amy-medium.onnx",
        config_path  = "models/piper/en_US-amy-medium.onnx.json",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "piper_lessac",
        provider     = "piper",
        display_name = "Piper — Lessac (en_US/lessac/high)",
        description  = "High-quality American male voice. Fully offline.",
        voice_id     = "en_US-lessac-high",
        model_path   = "models/piper/en_US-lessac-high.onnx",
        config_path  = "models/piper/en_US-lessac-high.onnx.json",
        speed        = 1.0,
        language     = "en-us",
    ),
    VoiceProfile(
        id           = "piper_jenny",
        provider     = "piper",
        display_name = "Piper — Jenny (en_GB/jenny_dioco/medium)",
        description  = "British female voice from Piper.",
        voice_id     = "en_GB-jenny_dioco-medium",
        model_path   = "models/piper/en_GB-jenny_dioco-medium.onnx",
        config_path  = "models/piper/en_GB-jenny_dioco-medium.onnx.json",
        speed        = 1.0,
        language     = "en-gb",
    ),
)


# ===========================================================================
# ── Chatterbox TTS profiles (reference-audio-based voice conditioning)
#    Requires: reference_audio (.wav, 5–15 seconds of clean speech)
#    Install:  pip install chatterbox-tts
#    Notes:    model_path is optional if the engine auto-downloads weights.
# ===========================================================================
_register(
    VoiceProfile(
        id              = "chatterbox_custom",
        provider        = "chatterbox",
        display_name    = "Chatterbox — Custom Clone",
        description     = "Voice clone conditioned on a custom reference WAV.",
        reference_audio = "voices/custom_speaker.wav",
        speed           = 1.0,
        language        = "en-us",
        extra           = {"exaggeration": 0.5, "cfg_weight": 0.5},
    ),
    VoiceProfile(
        id              = "chatterbox_professional",
        provider        = "chatterbox",
        display_name    = "Chatterbox — Professional Male",
        description     = "Voice conditioned on a professional-sounding male reference.",
        reference_audio = "voices/professional_male.wav",
        speed           = 1.0,
        language        = "en-us",
        extra           = {"exaggeration": 0.3, "cfg_weight": 0.5},
    ),
)


# ===========================================================================
# ── XTTS / Coqui profiles (reference-audio-based voice cloning)
#    Requires: model_path (XTTS v2 folder) + reference_audio (.wav)
#    Install:  pip install TTS
#    Notes:    Download XTTS v2 weights via `tts --model tts_models/multilingual/multi-dataset/xtts_v2`
# ===========================================================================
_register(
    VoiceProfile(
        id              = "xtts_custom",
        provider        = "xtts",
        display_name    = "XTTS v2 — Custom Clone",
        description     = "Multilingual voice clone using Coqui XTTS v2.",
        reference_audio = "voices/custom_speaker.wav",
        model_path      = "models/xtts_v2/",
        speed           = 1.0,
        language        = "en",
        extra           = {"temperature": 0.7, "length_penalty": 1.0},
    ),
)


# ===========================================================================
# ── Qwen3-TTS profiles (Alibaba open-source)
#    Requires: model_path or voice_id depending on serving approach
#    Notes:    Stub — update extra fields once the engine wrapper is built.
# ===========================================================================
_register(
    VoiceProfile(
        id           = "qwen3_default",
        provider     = "qwen3",
        display_name = "Qwen3-TTS — Default",
        description  = "Alibaba Qwen3-TTS open-source model, default speaker.",
        voice_id     = "default",
        model_path   = "models/qwen3_tts/",
        speed        = 1.0,
        language     = "en",
    ),
)


# ===========================================================================
# ── Public API
# ===========================================================================

_DEFAULT_PROFILE_ID = "kokoro_heart"


def get_voice_profile(profile_id: str) -> VoiceProfile:
    """Return a VoiceProfile by its id.

    Raises KeyError with a helpful message listing available profiles if not found.
    """
    if profile_id not in _PROFILES:
        available = ", ".join(sorted(_PROFILES))
        raise KeyError(
            f"Unknown voice profile {profile_id!r}. "
            f"Available profiles: {available}"
        )
    return _PROFILES[profile_id]


def get_active_voice_profile() -> VoiceProfile:
    """Return the active profile, driven by the VOICE_PROFILE env var.

    Defaults to 'kokoro_heart' if the variable is unset or empty.
    Reads the env var on every call so hot-swapping works without restart
    (when the calling code also re-reads the profile before each TTS call).
    """
    profile_id = os.getenv("VOICE_PROFILE", _DEFAULT_PROFILE_ID).strip()
    return get_voice_profile(profile_id)


def list_voice_profiles() -> list[VoiceProfile]:
    """Return all registered profiles sorted by provider then id."""
    return sorted(_PROFILES.values(), key=lambda p: (p.provider, p.id))
