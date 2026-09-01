"""
Renderer — the only module in this prototype that touches FFmpeg.

This is a REAL renderer, not a mock: it produces an actual playable .mp4 at
the resolution appropriate for the target platform, with:
  - per-scene visuals (user photo, framed with a rounded card; or a
    generated template/placeholder card for template/stock/ai scenes)
  - consistent typography/palette across every scene (pulled straight off
    the StoryboardPlan, never re-decided per scene)
  - automatic transitions (crossfade / slide / zoom) between scenes, via
    FFmpeg's `xfade` filter
  - a synthesized placeholder music bed (see music_library.py) with
    fade-in/fade-out

Where real production pieces plug in later (see ARCHITECTURE.md section 7):
  - Face-aware cropping: `_compose_user_media_frame` does a center-crop
    today; the TODO below marks exactly where a face-detection bounding
    box would change the crop rectangle.
  - Stock/AI imagery: `_compose_placeholder_frame` draws a clearly-labeled
    placeholder card for STOCK/AI_GENERATED scenes; a real implementation
    would fetch/generate the actual image here instead and skip the label.
  - Narration/TTS: `Scene.narration` text already exists; a TTS engine
    would synthesize per-scene audio here and the mixing step below would
    duck the music under it.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import Platform, RenderOutputMeta, StoryboardPlan, TransitionType, VisualSourceType
from .music_library import MusicLibrary

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
    TransitionType.ZOOM: ("zoomin", 0.5),
}

_FPS = 24

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
    ) -> RenderOutputMeta:
        job_dir = Path(job_dir)
        tmp_dir = job_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        base_w, base_h = _BASE_RESOLUTION[storyboard.intent.platform]
        width = _even(int(base_w * resolution_scale))
        height = _even(int(base_h * resolution_scale))

        clip_paths, durations, transitions = self._render_scene_clips(
            storyboard, tmp_dir, width, height, variant_seed
        )
        silent_video = tmp_dir / "silent.mp4"
        total_duration = self._concat_with_transitions(
            clip_paths, durations, transitions, silent_video
        )

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

    # -- stage 1: one static frame + clip per scene -------------------------

    def _render_scene_clips(self, storyboard, tmp_dir: Path, width: int, height: int, variant_seed: int = 0):
        scenes = storyboard.scenes
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

        clip_paths = []
        for scene, render_duration in zip(scenes, render_durations):
            frame = self._compose_frame(scene, storyboard, width, height, variant_seed)
            frame_path = tmp_dir / f"scene_{scene.index:02d}.png"
            frame.save(frame_path)

            clip_path = tmp_dir / f"scene_{scene.index:02d}.mp4"
            _run(
                [
                    FFMPEG_BIN, "-y", "-loglevel", "error",
                    "-loop", "1", "-i", str(frame_path),
                    "-t", f"{render_duration:.3f}",
                    "-vf", f"fps={_FPS},format=yuv420p",
                    "-c:v", "libx264",
                    str(clip_path),
                ]
            )
            clip_paths.append(clip_path)
        return clip_paths, render_durations, transitions

    def _compose_frame(self, scene, storyboard: StoryboardPlan, width: int, height: int, variant_seed: int = 0) -> Image.Image:
        visual = scene.visual
        if visual.source_type == VisualSourceType.USER_MEDIA and visual.user_media_ref:
            img = self._compose_user_media_frame(visual, storyboard, width, height)
        else:
            img = self._compose_placeholder_frame(scene, storyboard, width, height, variant_seed)

        self._draw_caption(img, scene.caption, storyboard, width, height)
        return img

    def _compose_user_media_frame(self, visual, storyboard: StoryboardPlan, width: int, height: int) -> Image.Image:
        palette = storyboard.palette
        bg = Image.new("RGB", (width, height), _hex_to_rgb(palette.secondary))

        source_path = Path(visual.user_media_ref)
        photo = self._load_still_from_media(source_path)

        # Framed "card": the photo fills a rounded-rect inset with a small
        # margin, rather than bleeding to the frame edges — this is the
        # "auto-crop and frame tastefully (rounded corners, simple border)"
        # requirement from the brief.
        margin = int(min(width, height) * 0.045)
        card_w, card_h = width - 2 * margin, height - 2 * margin
        radius = int(min(card_w, card_h) * 0.06)

        # TODO(face-aware crop): a real implementation would run a face
        # detector on `photo` here and shift/expand this cover-crop box so
        # faces stay centered instead of a plain geometric center-crop.
        cropped = _cover_crop(photo, card_w, card_h)

        mask = Image.new("L", (card_w, card_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, card_w, card_h], radius=radius, fill=255)
        bg.paste(cropped, (margin, margin), mask)

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

    def _compose_placeholder_frame(self, scene, storyboard: StoryboardPlan, width: int, height: int, variant_seed: int = 0) -> Image.Image:
        palette = storyboard.palette
        top = _hex_to_rgb(palette.primary)
        bottom = _hex_to_rgb(palette.secondary)
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y in range(height):
            t = y / max(1, height - 1)
            row_color = tuple(int(top[c] * (1 - t) + bottom[c] * t) for c in range(3))
            for x in range(0, width, 4):  # step of 4: cheap perf win, banding is imperceptible
                for dx in range(4):
                    if x + dx < width:
                        pixels[x + dx, y] = row_color

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
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        # This *is* the plug-in point for real stock/AI imagery: in
        # production, STOCK/AI_GENERATED scenes fetch/generate a real photo
        # here instead of a gradient card, and the small source tag below is
        # removed. Kept visible in the prototype so it's obvious, in any
        # rendered sample, which scenes are template vs. would-be real
        # imagery.
        if scene.visual.source_type != VisualSourceType.TEMPLATE:
            tag = "STOCK PHOTO (placeholder)" if scene.visual.source_type == VisualSourceType.STOCK else "AI IMAGE (placeholder)"
            draw = ImageDraw.Draw(img)
            font = self._font(storyboard.font_family, int(height * 0.018))
            draw.text((int(width * 0.03), int(height * 0.03)), tag, font=font, fill=(*accent, 255)[:3])

        return img

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
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)]
        )
        _run(cmd)
        return running_total

    # -- stage 3: synthesize + mux a placeholder music bed -------------------

    def _mux_with_music(self, silent_video: Path, storyboard: StoryboardPlan, duration: float,
                         output_path: Path, variant_seed: int) -> None:
        track = self._music_library.pick_track(storyboard.music_mood, variant_seed=variant_seed)
        freqs = track.synth_recipe["chord_freqs_hz"]
        amplitude = track.synth_recipe["amplitude"]

        fade_len = min(1.5, duration / 4) if duration > 0.5 else duration / 2

        sine_inputs = []
        for freq in freqs:
            sine_inputs += ["-f", "lavfi", "-i", f"sine=frequency={freq:.3f}:duration={duration:.3f}:sample_rate=44100"]

        mix_labels = "".join(f"[{i}:a]" for i in range(1, len(freqs) + 1))
        filter_complex = (
            f"{mix_labels}amix=inputs={len(freqs)}:duration=first:dropout_transition=0,"
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
    return resized.crop((left, top, left + target_w, top + target_h))
