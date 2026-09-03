"""
voiceover.py — real, spoken narration for each scene, synthesized from the
`Scene.narration` text that StoryboardGenerator already writes (see its
module docstring, and the TODO this module resolves in renderer.py:
"Narration/TTS: Scene.narration text already exists; a TTS engine would
synthesize per-scene audio here and the mixing step below would duck the
music under it.").

Two tiers, both free and keyless — no signup, no API key, no per-request
cost, the same hard constraint every other part of this app is built
around:

  1. Microsoft Edge's online neural voices, via the `edge-tts` package.
     This is the same real, natural-sounding text-to-speech Edge's own
     "Read aloud" feature uses, exposed as a public, keyless streaming
     endpoint. When it works, the result sounds like an actual human
     narrator, not a robot — this is the primary path.
  2. `espeak-ng`, a fully offline command-line synthesizer, as the
     fallback when tier 1 fails for any reason (no outbound path to
     Microsoft's endpoint from this particular host, a timeout, a
     transient error). It sounds noticeably more mechanical, but it has
     zero network dependency, so a render still gets *a* voiceover
     instead of silently losing the feature.

Design constraints (same contract as stock_media.py):
  - MUST NEVER raise. Every failure mode returns None and the caller
    (Renderer) simply renders that scene's narration as silence — the
    same "an optional enhancement can degrade, but must never break the
    render" rule stock photos already follow.
  - MUST be time-bounded. A hung network call must not stall a render
    indefinitely on an already resource-constrained free-tier host.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .models import Tone

try:
    import edge_tts
except ImportError:  # pragma: no cover - always installed via requirements.txt
    edge_tts = None

# One natural-sounding Microsoft neural voice per tone, so the narrator's
# voice matches the mood of the video instead of being one-size-fits-all.
_VOICE_BY_TONE: dict[Tone, str] = {
    Tone.FUN: "en-US-AriaNeural",
    Tone.EMOTIONAL: "en-US-JennyNeural",
    Tone.ELEGANT: "en-US-ElizabethNeural",
    Tone.CALM: "en-US-MichelleNeural",
    Tone.BOLD: "en-US-GuyNeural",
    Tone.WARM: "en-US-AmberNeural",
    Tone.PROFESSIONAL: "en-US-EricNeural",
}
_DEFAULT_VOICE = "en-US-AriaNeural"

_EDGE_TTS_TIMEOUT_SECONDS = 10.0
_ESPEAK_TIMEOUT_SECONDS = 15.0
_MIN_VALID_AUDIO_BYTES = 500  # guards against a 0-byte/near-empty "success"


def voice_for_tone(tone: Tone) -> str:
    return _VOICE_BY_TONE.get(tone, _DEFAULT_VOICE)


def synthesize_narration(text: str, out_stem: Path, voice: str) -> Optional[Path]:
    """Best-effort: synthesize spoken audio for `text`, writing it next to
    `out_stem` (the caller supplies a path *without* a meaningful suffix,
    since which tier succeeds determines the real file format). Returns the
    actual audio file path on success, or None if neither tier produced
    usable audio — this function never raises."""
    text = (text or "").strip()
    if not text:
        return None

    out_stem = Path(out_stem)

    mp3_path = out_stem.with_suffix(".mp3")
    if _synthesize_with_edge_tts(text, mp3_path, voice):
        return mp3_path

    wav_path = out_stem.with_suffix(".wav")
    if _synthesize_with_espeak(text, wav_path):
        return wav_path

    return None


def _synthesize_with_edge_tts(text: str, out_path: Path, voice: str) -> bool:
    if edge_tts is None:
        return False
    try:
        asyncio.run(
            asyncio.wait_for(_edge_tts_save(text, out_path, voice), timeout=_EDGE_TTS_TIMEOUT_SECONDS)
        )
    except Exception:
        # A failed/interrupted connection can still leave a 0-byte (or
        # truncated) file behind from opening the output for write; clear
        # it out so it can't be mistaken for a real result and so the
        # per-job tmp dir doesn't accumulate dead files across scenes.
        out_path.unlink(missing_ok=True)
        return False
    ok = out_path.exists() and out_path.stat().st_size > _MIN_VALID_AUDIO_BYTES
    if not ok:
        out_path.unlink(missing_ok=True)
    return ok


async def _edge_tts_save(text: str, out_path: Path, voice: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def _synthesize_with_espeak(text: str, wav_path: Path) -> bool:
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak_bin:
        return False
    try:
        result = subprocess.run(
            [espeak_bin, "-v", "en-us", "-s", "165", "-w", str(wav_path), text],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=_ESPEAK_TIMEOUT_SECONDS,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return wav_path.exists() and wav_path.stat().st_size > _MIN_VALID_AUDIO_BYTES
