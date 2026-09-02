"""
Renderer — the only module in this prototype that touches FFmpeg.

This is a REAL renderer, not a mock: it produces an actual playable .mp4 at
the resolution appropriate for the target platform, with:
  - per-scene visuals: the user's own photo (framed as a keepsake card), or
    a real fetched stock photo (full-bleed, cinematic) when the user didn't
    upload media — falling back to a generated gradient card only if a
    stock photo genuinely couldn't be fetched (offline, no results, etc.)
  - a slow "Ken Burns" zoom on every scene (via ffmpeg's zoompan filter) so
    the output reads as video, not a slideshow of static cards
  - consistent typography/palette across every scene (pulled straight off
    the StoryboardPlan, never re-decided per scene)
  - automatic transitions (crossfade / slide / zoom) between scenes, via
    FFmpeg's `xfade` filter
  - a synthesized two-chord music bed (see music_library.py) with a gentle
    tempo-synced pulse, fade-in/fade-out

Where real production pieces plug in later (see ARCHITECTURE.md section 7):
  - Face-aware cropping: `_compose_user_media_frame` does a center-crop
    today; the TODO below marks exactly where a face-detection bounding
    box would change the crop rectangle.
  - Narration/TTS: `Scene.narration` text already exists; a TTS engine
    would synthesize per-scene audio here and the mixing step below would
    duck the music under it.
  - Real stock photo attribution: stock_media.py deliberately keeps this
    abstraction narrow (bytes in, bytes out) — a version of this app that
    shows these videos publicly should track and surface each photo's
    attribution/license per Openverse's terms, which this prototype does
    not yet render on-screen.

Memory notes (this runs on memory-constrained free-tier hosts):
  - Every PIL image is explicitly closed/deleted and `gc.collect()` +
    `malloc_trim` are called between scenes and at the end of a render, so
    a long-lived server process gives memory back to the OS between jobs
    instead of slowly ratcheting up.
  - Downloaded stock photos are thumbnailed down immediately after opening,
    regardless of their original size, so one unusually large source photo
    can't spike memory.
  - FFmpeg encodes use `-preset ultrafast -threads 1`: on a 0.1-vCPU host,
    a slower/higher-quality preset buys negligible quality for real memory
    and wall-clock cost.
"""
from __future__ import annotations

import ctypes
import gc
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from .models import Platform, RenderOutputMeta, StoryboardPlan, TransitionType, VisualSourceType
from .music_library import MusicLibrary
from .stock_media import fetch_stock_photo_file

# --------------------------------------------------------------------------
# Platform -> output resolution (full res). Tests/dev use `resolution_scale`
# to render smaller/faster while keeping the exact same aspect ratio.
# --------------------------------------------------------------------------

_BASE_RESOLUTION = {
    Platform.VERTICAL: (1080, 1920),
    Platform.HORIZONTAL: (1920, 1080),
    Platform.SQUARE: (1080, 1080),
}

# Fonts are bundled with the app (backend/app/assets/fonts) rather than
# referenced by OS-specific system paths, so rendering is identical whether
# this runs in the Linux dev container or on a user's Mac.
_FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
_FONT_FILE_MAP = {
    "Poppins-Bold": str(_FONTS_DIR / "DejaVuSans-Bold.ttf"),
    "PlayfairDisplay-Regular": str(_FONTS_DIR / "DejaVuSerif.ttf"),
    "Montserrat-Black": str(_FONTS_DIR / "LiberationSans-Bold.ttf"),
    "Quicksand-Medium": str(_FONTS_DIR / "DejaVuSans.ttf"),
    "Lato-Regular": str(_FONTS_DIR / "DejaVuSans.ttf"),
    "Inter-Medium": str(_FONTS_DIR / "LiberationSans-Regular.ttf"),
}
_FALLBACK_FONT = str(_FONTS_DIR / "DejaVuSans-Bold.ttf")

# FFmpeg/ffprobe are looked up on PATH by default, but can be overridden —
# e.g. to point at a bundled binary — via environment variables. This is
# what lets the same code run against a system install (Linux dev/CI) or a
# standalone binary dropped next to the app (see run.sh / Start_AutoVideo.
# command), with no code change either way.
FFMPEG_BIN = os.environ.get("AUTOVIDEO_FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("AUTOVIDEO_FFPROBE_BIN", "ffprobe")

_XFADE_BY_TRANSITION = {
    TransitionType.CUT: ("fade", 0.15),
    TransitionType.CROSSFADE: ("fade", 0.6),
    TransitionType.SLIDE: ("slideleft", 0.5),
    # NOTE: xfade's "zoomin"/"zoomout" transitions only exist on very recent
    # ffmpeg builds (added long after the xfade filter itself) and are
    # missing on plenty of ffmpeg versions still shipped by current Linux
    # distros — including the exact ffmpeg apt pulls in this project's own
    # Dockerfile on at least one real deploy target tested. Using one would
    # crash every render that happens to pick a ZOOM transition on any host
    # with an older ffmpeg. "circleopen" gives a similar "reveal" feel and
    # has been supported since the xfade filter's original release.
    TransitionType.ZOOM: ("circleopen", 0.5),
}

_FPS = 24

# Ken Burns: every scene slowly zooms in from 1.0x to this factor over the
# scene's full duration. Small enough to feel gentle rather than gimmicky.
_KEN_BURNS_MAX_ZOOM = 1.09

# The bundled system fonts (DejaVu/Liberation) have no emoji glyphs, so an
# emoji in a caption would render as a "tofu" box. Strip pictographic
# characters from the *rendered* text only — the caption stored on the
# Scene (and shown anywhere else, e.g. a future TTS pass) keeps the emoji.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed ({' '.join(cmd)}):\n{result.stdout[-4000:]}")


def _release_memory() -> None:
    """Best-effort: force Python to garbage-collect and hand freed memory
    back to the OS. Long-lived Python processes doing heavy Pillow/ffmpeg
    work otherwise tend to keep fragmented memory resident even after it's
    no longer referenced, which is how a free-tier 512MB container can
    OOM on a *second* render even though no single moment held that much
    live data. `malloc_trim` is glibc-specific (Linux) and a no-op/harmless
    failure everywhere else (e.g. the macOS desktop build)."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


class Renderer:
    """See module docstring."""

    def __init__(self) -> None:
        self._music_library = MusicLibrary()

    def render(
        self,
        storyboard: StoryboardPlan,
        job_dir: Path,
        resolution_scale: float = 1.0,
        variant_seed: int = 0,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> RenderOutputMeta:
        job_dir = Path(job_dir)
        tmp_dir = job_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _notify = progress_cb or (lambda _msg: None)

        try:
            base_w, base_h = _BASE_RESOLUTION[storyboard.intent.platform]
            width = _even(int(base_w * resolution_scale))
            height = _even(int(base_h * resolution_scale))

            clip_paths, durations, transitions = self._render_scene_clips(
                storyboard, tmp_dir, width, height, variant_seed, _notify
            )
            _notify("Combining scenes...")
            silent_video = tmp_dir / "silent.mp4"
            total_duration = self._concat_with_transitions(
                clip_paths, durations, transitions, silent_video
            )

            _notify("Adding music...")
            output_path = job_dir / "output.mp4"
            self._mux_with_music(
                silent_video, storyboard, total_duration, output_path, variant_seed
            )

            file_size = output_path.stat().st_size
            return RenderOutputMeta(
                width=width,
                height=height,
                duration_seconds=round(total_duration, 2),
                file_size_bytes=file_size,
                codec="h264",
                path=str(output_path),
            )
        finally:
            _release_memory()

    # -- stage 1: one static frame + clip per scene -------------------------

    def _render_scene_clips(self, storyboard, tmp_dir: Path, width: int, height: int, variant_seed: int, notify) -> tuple[list[Path], list[float], list[TransitionType]]:
        scenes = storyboard.scenes
        n_scenes = len(scenes)
        transitions = [s.transition_in for s in scenes]

        # A crossfade between clip i and clip i+1 makes them overlap for the
        # transition's duration, which shortens the *total* concatenated
        # video by that amount at every junction. Left uncompensated, a
        # storyboard with many scenes (and therefore many transitions) can
        # render noticeably shorter than the requested length (caught by
        # the Phase 5 scenario-matrix test). Fix: render each scene's clip
        # a little longer than its storyboard duration — by exactly the
        # overlap its *outgoing* transition will consume — so the final,
        # post-crossfade duration matches the storyboard's target exactly.
        render_durations = [s.duration_seconds for s in scenes]
        for i in range(len(scenes) - 1):
            _, xfade_dur = _XFADE_BY_TRANSITION[transitions[i + 1]]
            render_durations[i] += xfade_dur

        stock_cache_dir = tmp_dir / "stock_cache"

        clip_paths = []
        for i, (scene, render_duration) in enumerate(zip(scenes, render_durations)):
            notify(f"Rendering scene {i + 1} of {n_scenes}...")

            # The background (photo/gradient) and the caption are composed
            # as two SEPARATE images. This matters once Ken Burns motion is
            # in play: the background is the thing that zooms, but the
            # caption overlay is composited on top of the zoomed video
            # afterwards, at a fixed size/position — otherwise the zoom
            # would also crop the caption's edges out of frame over time.
            background = self._compose_background(scene, storyboard, width, height, variant_seed, stock_cache_dir)
            bg_path = tmp_dir / f"scene_{scene.index:02d}_bg.png"
            background.save(bg_path)
            background.close()
            del background

            caption_overlay = self._compose_caption_overlay(scene.caption, storyboard, width, height)
            caption_path = tmp_dir / f"scene_{scene.index:02d}_caption.png"
            caption_overlay.save(caption_path)
            caption_overlay.close()
            del caption_overlay

            clip_path = tmp_dir / f"scene_{scene.index:02d}.mp4"
            n_frames = max(1, round(render_duration * _FPS))
            zoom_rate = (_KEN_BURNS_MAX_ZOOM - 1.0) / n_frames
            zoom_expr = f"min(zoom+{zoom_rate:.8f},{_KEN_BURNS_MAX_ZOOM})"
            filter_complex = (
                f"[0:v]zoompan=z='{zoom_expr}':d={n_frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={_FPS}[bg];"
                "[1:v]format=rgba[cap];"
                "[bg][cap]overlay=format=auto[ov];"
                "[ov]format=yuv420p[outv]"
            )
            _run(
                [
                    FFMPEG_BIN, "-y", "-loglevel", "error",
                    "-loop", "1", "-i", str(bg_path),
                    "-loop", "1", "-i", str(caption_path),
                    "-frames:v", str(n_frames),
                    "-filter_complex", filter_complex,
                    "-map", "[outv]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
                    str(clip_path),
                ]
            )
            clip_paths.append(clip_path)
            _release_memory()
        return clip_paths, render_durations, transitions

    def _compose_background(self, scene, storyboard: StoryboardPlan, width: int, height: int, variant_seed: int, stock_cache_dir: Path) -> Image.Image:
        visual = scene.visual
        if visual.source_type == VisualSourceType.USER_MEDIA and visual.user_media_ref:
            return self._compose_user_media_frame(visual, storyboard, width, height)
        return self._compose_stock_or_gradient_frame(scene, storyboard, width, height, variant_seed, stock_cache_dir)

    def _compose_caption_overlay(self, caption: str, storyboard: StoryboardPlan, width: int, height: int) -> Image.Image:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        self._draw_caption(overlay, caption, storyboard, width, height)
        return overlay

    def _compose_user_media_frame(self, visual, storyboard: StoryboardPlan, width: int, height: int) -> Image.Image:
        palette = storyboard.palette
        bg = Image.new("RGB", (width, height), _hex_to_rgb(palette.secondary))

        source_path = Path(visual.user_media_ref)
        photo = self._load_still_from_media(source_path)

        # Framed "card": the photo fills a rounded-rect inset with a small
        # margin, rather than bleeding to the frame edges — this is the
        # "auto-crop and frame tastefully (rounded corners, simple border)"
        # requirement from the brief. Personal photos get this treatment
        # (a keepsake); stock backdrops go full-bleed instead (cinematic) —
        # see `_compose_stock_or_gradient_frame`.
        margin = int(min(width, height) * 0.045)
        card_w, card_h = width - 2 * margin, height - 2 * margin
        radius = int(min(card_w, card_h) * 0.06)

        # TODO(face-aware crop): a real implementation would run a face
        # detector on `photo` here and shift/expand this cover-crop box so
        # faces stay centered instead of a plain geometric center-crop.
        cropped = _cover_crop(photo, card_w, card_h)
        photo.close()

        mask = Image.new("L", (card_w, card_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, card_w, card_h], radius=radius, fill=255)
        bg.paste(cropped, (margin, margin), mask)
        cropped.close()

        # Simple border, in the palette's accent color, framing the photo.
        draw = ImageDraw.Draw(bg)
        border_w = max(3, int(min(width, height) * 0.006))
        draw.rounded_rectangle(
            [margin, margin, margin + card_w, margin + card_h],
            radius=radius,
            outline=_hex_to_rgb(palette.accent),
            width=border_w,
        )
        return bg

    def _load_still_from_media(self, path: Path) -> Image.Image:
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".webm"}:
            # Short video clip: pull a representative still frame (~20% in)
            # for this prototype. A production renderer would use the full
            # motion clip instead of a single frame.
            still_path = path.with_suffix(".still.png")
            duration = 1.0
            try:
                probe = subprocess.run(
                    [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                    stdout=subprocess.PIPE, text=True,
                )
                duration = float(probe.stdout.strip())
            except (FileNotFoundError, ValueError):
                # ffprobe isn't bundled (only ffmpeg is, to keep the
                # standalone install small) — fall back to grabbing an
                # early frame rather than failing the whole render.
                duration = 1.0
            grab_at = max(0.1, duration * 0.2)
            _run([FFMPEG_BIN, "-y", "-loglevel", "error", "-ss", f"{grab_at:.2f}",
                  "-i", str(path), "-frames:v", "1", str(still_path)])
            return Image.open(still_path).convert("RGB")
        return Image.open(path).convert("RGB")

    def _compose_stock_or_gradient_frame(self, scene, storyboard: StoryboardPlan, width: int, height: int, variant_seed: int, stock_cache_dir: Path) -> Image.Image:
        """Full-bleed real stock photo when one can be fetched; otherwise
        the gradient-card fallback (offline, no results, corrupt image,
        etc. — this path must never fail the render)."""
        if scene.visual.source_type == VisualSourceType.STOCK and scene.visual.stock_query:
            photo = self._try_fetch_stock_photo(scene.visual.stock_query, stock_cache_dir)
            if photo is not None:
                framed = _cover_crop(photo, width, height)
                photo.close()
                # A subtle darkening gradient at the bottom keeps the
                # caption bar legible over any photo, without needing a
                # flat black bar the whole width of the frame.
                framed = framed.convert("RGB")
                self._apply_bottom_scrim(framed)
                return framed
        return self._compose_gradient_frame(scene, storyboard, width, height, variant_seed)

    def _try_fetch_stock_photo(self, query: str, cache_dir: Path) -> Optional[Image.Image]:
        try:
            path = fetch_stock_photo_file(query, cache_dir)
            if not path:
                return None
            img = Image.open(path)
            # Bound memory regardless of the source photo's actual size —
            # this runs before .convert(), which is what forces the full
            # decode into memory.
            img.thumbnail((1600, 1600), Image.LANCZOS)
            return img.convert("RGB")
        except Exception:
            return None

    def _apply_bottom_scrim(self, img: Image.Image) -> None:
        width, height = img.size
        scrim_h = int(height * 0.35)
        scrim = Image.new("L", (1, scrim_h), 0)
        pixels = scrim.load()
        for y in range(scrim_h):
            pixels[0, y] = int(140 * (y / max(1, scrim_h - 1)))
        scrim = scrim.resize((width, scrim_h), Image.BILINEAR)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        black = Image.new("RGBA", (width, scrim_h), (0, 0, 0, 255))
        overlay.paste(black, (0, height - scrim_h), scrim)
        composited = Image.alpha_composite(img.convert("RGBA"), overlay)
        img.paste(composited.convert("RGB"))
        overlay.close()
        black.close()
        scrim.close()
        composited.close()

    def _compose_gradient_frame(self, scene, storyboard: StoryboardPlan, width: int, height: int, variant_seed: int = 0) -> Image.Image:
        palette = storyboard.palette
        top = _hex_to_rgb(palette.primary)
        bottom = _hex_to_rgb(palette.secondary)
        # Build the vertical gradient as a 1px-wide column, then let PIL's
        # (C-level) resize stretch it to full width — avoids a
        # width*height Python-level loop.
        grad = Image.new("RGB", (1, height))
        grad_pixels = grad.load()
        for y in range(height):
            t = y / max(1, height - 1)
            grad_pixels[0, y] = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
        img = grad.resize((width, height), Image.NEAREST)
        grad.close()

        # Subtle decorative texture so a template card doesn't look like a
        # flat swatch: a few soft translucent circles in the accent color.
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        accent = _hex_to_rgb(palette.accent)
        for i in range(3):
            cx = int(width * (0.2 + 0.3 * ((i + variant_seed) % 3)))
            cy = int(height * (0.15 + 0.25 * ((i + scene.index + variant_seed) % 3)))
            r = int(min(width, height) * 0.18)
            odraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*accent, 28))
        composited = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        img.close()
        overlay.close()
        return composited

    def _draw_caption(self, img: Image.Image, caption: str, storyboard: StoryboardPlan, width: int, height: int) -> None:
        palette = storyboard.palette
        font_size = int(height * 0.052)
        font = self._font(storyboard.font_family, font_size)

        draw = ImageDraw.Draw(img, "RGBA")
        caption = _strip_emoji(caption)
        max_text_width = width * 0.86
        # Real pixel-width wrapping (not a character-count guess): keeps
        # long captions and bold/serif fonts from overflowing the frame.
        lines = _wrap_text_by_pixels(draw, caption, font, max_text_width)
        # If a single word is still wider than the frame (e.g. a very long
        # word at a very narrow resolution), shrink the font until it fits.
        while any(draw.textlength(line, font=font) > max_text_width for line in lines) and font_size > 14:
            font_size = int(font_size * 0.9)
            font = self._font(storyboard.font_family, font_size)
            lines = _wrap_text_by_pixels(draw, caption, font, max_text_width)

        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
        total_text_h = sum(line_heights) + (len(lines) - 1) * int(font_size * 0.25)

        bar_h = total_text_h + int(font_size * 1.3)
        bar_top = height - bar_h - int(height * 0.06)
        # Semi-transparent bar behind the caption for legibility over any
        # background (photo or gradient) — consistent treatment every scene.
        draw.rectangle([0, bar_top, width, bar_top + bar_h], fill=(0, 0, 0, 110))

        y = bar_top + int(font_size * 0.4)
        text_color = _hex_to_rgb(palette.text_on_dark)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            x = (width - line_w) / 2
            draw.text((x, y), line, font=font, fill=text_color)
            y += bbox[3] + int(font_size * 0.25)

    def _font(self, font_family: str, size: int) -> ImageFont.FreeTypeFont:
        path = _FONT_FILE_MAP.get(font_family, _FALLBACK_FONT)
        return ImageFont.truetype(path, size)

    # -- stage 2: concat with xfade transitions ------------------------------

    def _concat_with_transitions(self, clip_paths, durations, transitions, output_path: Path) -> float:
        if len(clip_paths) == 1:
            _run([FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(clip_paths[0]), "-c", "copy", str(output_path)])
            return durations[0]

        inputs = []
        for p in clip_paths:
            inputs += ["-i", str(p)]

        filter_parts = []
        prev_label = "[0:v]"
        running_total = durations[0]
        for i in range(1, len(clip_paths)):
            xfade_type, xfade_dur = _XFADE_BY_TRANSITION[transitions[i]]
            xfade_dur = min(xfade_dur, durations[i] * 0.9, running_total * 0.9)
            xfade_dur = max(xfade_dur, 0.05)
            offset = max(0.0, running_total - xfade_dur)
            out_label = f"[v{i}]"
            filter_parts.append(
                f"{prev_label}[{i}:v]xfade=transition={xfade_type}:duration={xfade_dur:.3f}:offset={offset:.3f}{out_label}"
            )
            running_total = running_total + durations[i] - xfade_dur
            prev_label = out_label

        filter_complex = ";".join(filter_parts)
        cmd = (
            [FFMPEG_BIN, "-y", "-loglevel", "error"]
            + inputs
            + ["-filter_complex", filter_complex, "-map", prev_label,
               "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
               "-pix_fmt", "yuv420p", str(output_path)]
        )
        _run(cmd)
        return running_total

    # -- stage 3: synthesize + mux a two-chord music bed ---------------------

    def _mux_with_music(self, silent_video: Path, storyboard: StoryboardPlan, duration: float,
                         output_path: Path, variant_seed: int) -> None:
        track = self._music_library.pick_track(storyboard.music_mood, variant_seed=variant_seed)
        chord_a_freqs = track.synth_recipe["chord_freqs_hz"]
        chord_b_freqs = track.synth_recipe["chord_b_freqs_hz"]
        amplitude = track.synth_recipe["amplitude"]
        tempo_bpm = track.tempo_bpm

        fade_len = min(1.5, duration / 4) if duration > 0.5 else duration / 2
        d1 = max(0.2, duration / 2)
        d2 = max(0.2, duration - d1)

        sine_inputs = []
        for freq in chord_a_freqs:
            sine_inputs += ["-f", "lavfi", "-i", f"sine=frequency={freq:.3f}:duration={d1:.3f}:sample_rate=44100"]
        for freq in chord_b_freqs:
            sine_inputs += ["-f", "lavfi", "-i", f"sine=frequency={freq:.3f}:duration={d2:.3f}:sample_rate=44100"]

        n_a = len(chord_a_freqs)
        n_b = len(chord_b_freqs)
        mix_a_labels = "".join(f"[{i}:a]" for i in range(1, 1 + n_a))
        mix_b_labels = "".join(f"[{i}:a]" for i in range(1 + n_a, 1 + n_a + n_b))

        # A gentle amplitude pulse timed to the track's own tempo — the
        # difference between a flat, droning sine chord and something that
        # at least *feels* like it has a beat. Kept shallow (depth 0.25) so
        # it reads as a soft "breathing" pulse, not a tremolo effect.
        pulse_hz = max(0.5, (tempo_bpm / 60.0) / 2)

        filter_complex = (
            f"{mix_a_labels}amix=inputs={n_a}:duration=first[a_chordA];"
            f"{mix_b_labels}amix=inputs={n_b}:duration=first[a_chordB];"
            "[a_chordA][a_chordB]concat=n=2:v=0:a=1[a_concat];"
            f"[a_concat]tremolo=f={pulse_hz:.3f}:d=0.25,"
            f"volume={amplitude},"
            f"afade=t=in:st=0:d={fade_len:.2f},"
            f"afade=t=out:st={max(0.0, duration - fade_len):.2f}:d={fade_len:.2f}"
            "[aout]"
        )

        cmd = (
            [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(silent_video)]
            + sine_inputs
            + [
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(output_path),
            ]
        )
        _run(cmd)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _wrap_text_by_pixels(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """Word-wrap `text` so every line's rendered pixel width fits max_width,
    measuring with the actual font metrics rather than an average
    characters-per-line guess (bold/serif faces are wide enough that a
    guess reliably overflows narrow, vertical-video frames)."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _cover_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize+center-crop `img` to exactly (target_w, target_h), covering
    the whole target box (like CSS `object-fit: cover`)."""
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(math.ceil(new_h * src_ratio))
    else:
        new_w = target_w
        new_h = int(math.ceil(new_w / src_ratio))

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    if resized is not cropped:
        resized.close()
    return cropped
