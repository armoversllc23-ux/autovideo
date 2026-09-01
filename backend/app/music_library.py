"""
MusicLibrary — placeholder "royalty-free music" catalog.

Production version: this same `pick_track(mood) -> MusicTrack` contract
would call a licensed royalty-free API (Epidemic Sound, Artlist, etc.) or a
generative-music model and return a real audio file reference. For this
prototype, `synth_recipe` describes a simple triad (chord) of sine-wave
tones that the Renderer turns into an actual audio bed with FFmpeg — real,
audible, royalty-free-by-construction audio, without depending on any
external asset library or network access.
"""
from __future__ import annotations

from .models import MusicMood, MusicTrack

_SOFT_TAGS = {"gentle", "reflective", "ambient", "slow", "sparse", "piano"}


def _note(semitones_from_a3: float) -> float:
    """A3 = 220 Hz; equal-tempered semitone offset -> frequency."""
    return 220.0 * (2 ** (semitones_from_a3 / 12))


class MusicLibrary:
    """Deterministic mood -> placeholder track resolver. See module docstring."""

    def pick_track(self, mood: MusicMood, variant_seed: int = 0) -> MusicTrack:
        tempo_bpm = sum(mood.tempo_bpm_range) // 2
        is_soft = any(tag in _SOFT_TAGS for tag in mood.tags)

        # A minor triad for soft/reflective moods (root, minor third, fifth);
        # a major triad, one octave up, for upbeat moods (brighter register).
        if is_soft:
            root_offset = 0 + variant_seed  # slight per-variant detune for "Try a different version"
            freqs = [_note(root_offset), _note(root_offset + 3), _note(root_offset + 7)]
            amplitude = 0.12
        else:
            root_offset = 12 + variant_seed
            freqs = [_note(root_offset), _note(root_offset + 4), _note(root_offset + 7)]
            amplitude = 0.16

        track_id = f"placeholder-{'-'.join(mood.tags)}-{tempo_bpm}bpm-v{variant_seed}"
        return MusicTrack(
            track_id=track_id,
            tags=list(mood.tags),
            tempo_bpm=tempo_bpm,
            synth_recipe={
                "waveform": "sine",
                "chord_freqs_hz": freqs,
                "amplitude": amplitude,
            },
        )
